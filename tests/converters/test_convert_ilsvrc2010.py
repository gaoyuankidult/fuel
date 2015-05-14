import io
import logging
import os
import tarfile
from tempfile import NamedTemporaryFile
import unittest
import gzip

import numpy
from PIL import Image
from six.moves import xrange
import zmq

from fuel.server import recv_arrays, send_arrays
from fuel.utils.logging import ZMQLoggingHandler
from fuel.converters.ilsvrc2010 import (HasZMQProcessLogger,
                                        TrainSetVentilator, TrainSetWorker,
                                        TrainSetSink)


class MockH5PYData(object):
    def __init__(self, shape, dtype):
        self.data = numpy.empty(shape, dtype)
        self.dims = MockH5PYDims(len(shape))

    def __setitem__(self, where, what):
        self.data[where] = what


class MockH5PYFile(dict):
    def __init__(self):
        self.attrs = {}

    def create_dataset(self, name, shape, dtype):
        self[name] = MockH5PYData(shape, dtype)


class MockH5PYDim(object):
    def __init__(self, dims):
        self.dims = dims
        self.scales = []

    def attach_scale(self, dataset):
        # I think this is necessary for it to be valid?
        assert dataset in self.dims.scales.values()
        self.scales.append(dataset)


class MockH5PYDims(object):
    def __init__(self, ndim):
        self._dims = [MockH5PYDim(self) for _ in xrange(ndim)]
        self.scales = {}

    def create_scale(self, dataset, name):
        self.scales[name] = dataset


def create_jpeg_data(image):
    """Create a JPEG in memory.

    Parameters
    ----------
    image : ndarray, 3-dimensional
        Array data representing the image to save. Mode ('L', 'RGB',
        'CMYK') will be determined from the last (third) axis.

    Returns
    -------
    jpeg_data : bytes
        The image encoded as a JPEG, returned as raw bytes.

    """
    if image.shape[-1] == 1:
        mode = 'L'
    elif image.shape[-1] == 3:
        mode = 'RGB'
    elif image.shape[-1] == 4:
        mode = 'CMYK'
    else:
        raise ValueError("invalid shape")
    pil_image = Image.frombytes(mode=mode, size=image.shape[:2],
                                data=image.tobytes())
    jpeg_data = io.BytesIO()
    pil_image.save(jpeg_data, format='JPEG')
    return jpeg_data.getvalue()


def create_fake_jpeg_tar(seed, min_num_images=5, max_num_images=50,
                         min_size=20, size_range=30, filenames=None,
                         random=True):
    """Create a TAR file of ranodmly generated JPEG files.

    Parameters
    ----------
    seed : int or sequence
        Seed for a `numpy.random.RandomState`.
    min_num_images : int, optional
        The minimum number of images to put in the TAR file.
    max_num_images : int, optional
        The maximum number of images to put in the TAR file.
    min_size : int, optional
        The minimum width and minimum height of each image.
    size_range : int, optional
        Maximum number of pixels added to `min_size` for image
        dimensions.
    filenames : list, optional
        If provided, use these filenames. Otherwise generate them
        randomly. Must be at least `max_num_images` long.
    random : bool, optional
        If `False`, substitute an image full of a single number,
        the order of that image in processing.

    Returns
    -------
    tar_data : bytes
        A TAR file represented as raw bytes, containing between
        `min_num_images` and `max_num_images` JPEG files (inclusive).

    Notes
    -----
    Randomly choose between RGB, L and CMYK mode images. Also randomly
    gzips JPEGs to simulate the idiotic distribution format of
    ILSVRC2010.

    """
    rng = numpy.random.RandomState(seed)
    images = []
    if filenames is None:
        files = []
    else:
        if len(filenames) < max_num_images:
            raise ValueError('need at least max_num_images = %d filenames' %
                             max_num_images)
        files = filenames
    for i in xrange(rng.random_integers(min_num_images, max_num_images)):
        if filenames is None:
            files.append('%x.JPEG' % abs(hash(str(i))))
        im = rng.random_integers(0, 255,
                                 size=(rng.random_integers(min_size,
                                                           min_size +
                                                           size_range),
                                       rng.random_integers(min_size,
                                                           min_size +
                                                           size_range),
                                       rng.random_integers(1, 4)))
        if not random:
            im *= 0
            im += i
        if im.shape[-1] == 2:
            im = im[:, :, :1]
        images.append(im)
    files = sorted(files)
    temp_tar = io.BytesIO()
    with tarfile.open(fileobj=temp_tar, mode='w') as tar:
        for fn, image in zip(files, images):
            try:
                with NamedTemporaryFile(mode='wb', suffix='.JPEG',
                                        delete=False) as f:
                    if rng.uniform() < 0.2:
                        gzip_data = io.BytesIO()
                        with gzip.GzipFile(mode='wb', fileobj=gzip_data) as gz:
                            gz.write(create_jpeg_data(image))
                        f.write(gzip_data.getvalue())
                    else:
                        f.write(create_jpeg_data(image))
                tar.add(f.name, arcname=fn)
            finally:
                os.remove(f.name)
    return temp_tar.getvalue(), files[:len(images)]


def create_fake_tar_of_tars(seed, num_inner_tars, *args, **kwargs):
    """Create a nested TAR of TARs of JPEGs.

    Parameters
    ----------
    seed : int or sequence
        Seed for a `numpy.random.RandomState`.
    num_inner_tars : int
        Number of TAR files to place inside.

    Returns
    -------
    tar_data : bytes
        A TAR file represented as raw bytes, TAR files of generated
        JPEGs.
    names : list
        Names of the inner TAR files.
    jpeg_names : list of lists
        A list of lists containing the names of JPEGs in each inner TAR.


    Notes
    -----
    Remainder of positional and keyword arguments are passed on to
    :func:`create_fake_tar_of_tars`.

    """
    rng = numpy.random.RandomState(seed)
    seeds = rng.random_integers(0, 500000, size=(num_inner_tars,))
    tars, fns = list(zip(*[create_fake_jpeg_tar(s, *args, **kwargs)
                           for s in seeds]))
    names = sorted(str(abs(hash(str(-i - 1)))) + '.tar'
                   for i, t in enumerate(tars))
    data = io.BytesIO()
    with tarfile.open(fileobj=data, mode='w') as outer:
        for tar, name in zip(tars, names):
            try:
                with NamedTemporaryFile(mode='wb', suffix='.tar',
                                        delete=False) as f:
                    f.write(tar)
                outer.add(f.name, arcname=name)
            finally:
                os.remove(f.name)
    return data.getvalue(), names, fns


def push_pull_socket_pair(context):
    pull = context.socket(zmq.PULL)
    pull_port = pull.bind_to_random_port('tcp://*')
    push = context.socket(zmq.PUSH)
    push.connect('tcp://localhost:{}'.format(pull_port))
    return push, pull


def test_has_zmq_process_logger():
    class DummyParent(object):
        def __init__(self, context, logging_port):
            self.context = context
            self.logging_port = logging_port
            self.initialized = False
            self.logger = logging.getLogger(__name__)

        def initialize_sockets(self):
            self.initialized = True

    class Dummy(HasZMQProcessLogger, DummyParent):
        pass
    try:
        context = zmq.Context()
        socket = context.socket(zmq.PULL)
        port = socket.bind_to_random_port('tcp://*')
        d = Dummy(context, port)
        d.initialize_sockets()
        assert d.initialized
        assert any(isinstance(h, ZMQLoggingHandler) for h in d.logger.handlers)
    finally:
        context.destroy()


class TestTrainSetVentilator(unittest.TestCase):
    def setUp(self):
        self.context = zmq.Context()
        tar, self.names, self.jpeg_names = create_fake_tar_of_tars(0, 5)
        self.nums = [len(j) for j in self.jpeg_names]
        self.vent = TrainSetVentilator(io.BytesIO(tar), logging_port=None)

    def tearDown(self):
        self.context.destroy()

    def test_send(self):
        push, pull = push_pull_socket_pair(self.context)
        self.vent.send(push, (555, 'random_stuff.tar', b'12345'))
        self.assertEqual(pull.recv_pyobj(), (555, 'random_stuff.tar'))
        self.assertEqual(pull.recv(), b'12345')

    def test_produce(self):
        for i, (inner, name, num_jpegs) in enumerate(zip(self.vent.produce(),
                                                         self.names,
                                                         self.nums)):
            j, inner_name, inner_tar = inner
            self.assertEqual(i, j)
            self.assertEqual(inner_name, name)
            with tarfile.open(fileobj=io.BytesIO(inner_tar)) as tar:
                self.assertEqual(len(tar.getmembers()), num_jpegs)


class TestTrainSetWorker(unittest.TestCase):
    def setUp(self):
        self.context = zmq.Context()
        self.data, self.names, self.jpeg_names = create_fake_tar_of_tars(1, 20)
        self.nums = [len(j) for j in self.jpeg_names]
        patch_filenames = ['train/' + f for f in sum(self.jpeg_names, [])[::5]]
        patch_data, _ = create_fake_jpeg_tar(0, min_num_images=20,
                                             max_num_images=20,
                                             filenames=patch_filenames,
                                             random=False)
        self.patch_filenames = dict(zip(patch_filenames,
                                        xrange(len(patch_filenames))))
        patch = io.BytesIO(patch_data)
        self.wnid_map = dict((b.split('.')[0], a) for a, b in
                             enumerate(self.names))
        self.worker = TrainSetWorker(patch, self.wnid_map, self.nums, 10, 2,
                                     None)
        print(type(self.wnid_map))

    def tearDown(self):
        self.context.destroy()

    def test_send(self):
        push, pull = push_pull_socket_pair(self.context)
        rng = numpy.random.RandomState(5)
        images = rng.normal(size=(5, 10, 15))
        filenames = numpy.array(['abcdef', 'ghijkl', 'mnopqr', 'stuvw',
                                 'xyz12'], dtype='S6')
        self.worker.send(push, (5, images, filenames))
        self.assertEqual(pull.recv_pyobj(), 5)
        r_images, r_filenames = recv_arrays(pull)
        numpy.testing.assert_equal(images, r_images)
        numpy.testing.assert_equal(filenames, r_filenames)

    def test_recv(self):
        push, pull = push_pull_socket_pair(self.context)
        push.send_pyobj((555, 'random_stuff.tar'))
        push.send(b'12345')
        received = self.worker.recv(pull)
        self.assertEqual(received[:2], (555, 'random_stuff.tar'))
        self.assertEqual(received[2].getvalue(), b'12345')

    def test_process(self):
        with tarfile.open(fileobj=io.BytesIO(self.data)) as tar:
            for i, inner in enumerate(tar):
                batch = (i, inner.name,
                         io.BytesIO(tar.extractfile(inner.name).read()))
                for result in self.worker.process(batch):
                    label, images, filenames = result
                    print(type(self.wnid_map))
                    self.assertEqual(self.wnid_map[inner.name.split('.')[0]],
                                     label)
                    for fn, im in zip(filenames, images):
                        if fn in self.patch_filenames:
                            value = self.patch_filenames[fn]
                            self.assertTrue((im == value).all())

    def test_handle_exception(self):
        raise unittest.SkipTest("TODO")


class TestTrainSetSink(unittest.TestCase):
    def setUp(self):
        raise unittest.SkipTest("TODO")

    def tearDown(self):
        raise unittest.SkipTest("TODO")

    def test_done(self):
        raise unittest.SkipTest("TODO")

    def test_process(self):
        raise unittest.SkipTest("TODO")

    def test_finalize(self):
        raise unittest.SkipTest("TODO")


class TestTrainSetManager(unittest.TestCase):
    def setUp(self):
        raise unittest.SkipTest("TODO")

    def tearDown(self):
        raise unittest.SkipTest("TODO")

    def test_wait(self):
        raise unittest.SkipTest("TODO")


def test_process_train_set():
    raise unittest.SkipTest("TODO")


def test_process_other_set():
    raise unittest.SkipTest("TODO")


def test_cropped_resized_images_from_tar():
    raise unittest.SkipTest("TODO")


def test_load_images_from_tar_or_patch():
    raise unittest.SkipTest("TODO")


def test_permutation_by_class():
    raise unittest.SkipTest("TODO")


def test_read_devkit():
    raise unittest.SkipTest("TODO")


def test_read_metadata():
    raise unittest.SkipTest("TODO")


def test_extract_patch_images():
    raise unittest.SkipTest("TODO")
