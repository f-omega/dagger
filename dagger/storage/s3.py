"""Store operation results in S3-compatible storage."""
from abc import ABC, abstractmethod
from io import TextIOWrapper, BytesIO
from typing import Optional, BinaryIO, Any, Union
import boto3
import hashlib
import logging

from . import KVStorage
from .. import util, repo
from ..repo import get_global_repository


log = logging.getLogger('dagger.storage.s3')


class S3Encoder(ABC):
    @abstractmethod
    def serialize(self, x: Any, out: BinaryIO):
        """Serialize the given data to a file-like object."""
        pass

    @abstractmethod
    def deserialize(self, inp: BinaryIO) -> Any:
        """Deserialize the given data."""
        pass


class JSONEncoder(S3Encoder):
    """Default S3Encoder that encodes everything via dagger's JSON Scheme.
    """

    def serialize(self, x: Any, out: BinaryIO):
        """Serialize the given value via JSON."""
        textio = TextIOWrapper(
            out,
            encoding='utf-8',
            write_through=True,
        )
        enc = util.JSONEncoder()
        util.dump_json(enc.typed(x), textio)
        textio.detach()

    def deserialize(self, inp: BinaryIO) -> Any:
        """Deserialize the given value via JSON."""
        textio = TextIOWrapper(
            inp,
            encoding='utf-8',
            write_through=True,
        )
        ret = util.load_json(textio)
        textio.detach()
        return ret

    def __json__(self, e: util.JSONEncoder):
        """Encode the object as JSON."""
        return {}

    def __fromjson__(self, d: Any, dec: util.JSONDecoder):
        """Do nothing."""
        pass

class S3Config(repo.RepositoryItem):
    __slots__ = ('_name', '_kwargs', '_client')

    def __init__(self, name, repo=None, **kwargs):
        self._name = name

        if repo is None:
            repo = get_global_repository()

        if 'client' in kwargs:
            self._client = kwargs['client']
        else:
            self._client = boto3.client('s3', **kwargs)

        repo.add(self)

    def get_item_dependencies(self):
        return []

    @property
    def item_name(self):
        return self._name

    @classmethod
    def get_item_type(self):
        return 's3_config'

    def get_client(self):
        return self._client

class S3Storage(KVStorage):
    """Store operation results in s3 compatible storage.

    All objects passed must be JSONable, or encodable using the custom
    encoder.
    """

    __slots__ = ('encoder', 'client', 'client_name', 'prefix', 'bucket',)

    encoder: S3Encoder
    client: Optional[S3Config]

    def __init__(self, *,
                 bucket: str,
                 client: Union[S3Config, str],
                 encoder: Optional[S3Encoder] = None,
                 prefix: str = ''):
        """Initialize the storage from an S3Config."""
        super().__init__(keysep='/')

        if encoder is None:
            encoder = JSONEncoder()

        self.encoder = encoder
        self.bucket = bucket
        self.prefix = prefix

        if isinstance(client, S3Config):
            self.client_name = client.item_name
            self.client = client
        else:
            self.client_name = client
            self.client = None

    def get_client(self):
        """Get the client, if we have one, or look it up in the repository."""
        if self.client is None:
            self.client = self._context.repo[S3Config.get_item_type(), self.client_name]
        return self.client.get_client()

    def _hash(self, b: BytesIO) -> str:
        """Returns the SHA-512 hash of the data."""
        return hashlib.sha512(b.getvalue()).hexdigest()

    def store_keyed_result(self, key: str, x: Any):
        qkey = f'{self.prefix}{key}'

        body = BytesIO()
        self.encoder.serialize(x, body)

        result = self.get_client().put_object(Bucket=self.bucket,
                                                     Key=qkey, Body=body.getvalue())
        log.debug("Stored result with hash %s (Etag=%s, VersionId=%s)",
                  key, result['ETag'], result.get('VersionId', None))

    def load_keyed_result(self, key: str):
        qkey = f'{self.prefix}{key}'
        try:
            result = self.get_client().get_object(Bucket=self.bucket, Key=qkey)
        except:
            raise KeyError(key)

        return self.encoder.deserialize(result['Body'])

    def __json__(self, enc: util.JSONEncoder):
        """Encode the S3 storage as JSON."""
        return {'config': self.client_name,
                'bucket': self.bucket,
                'prefix': self.prefix,
                'encoder': enc.typed(self.encoder),}

    def __fromjson__(self, d: Any, dec: util.JSONDecoder):
        """Decode the S3 storage as JSON."""
        super().__init__(keysep='/')
        self.encoder = d['encoder']
        self.bucket = d['bucket']
        self.prefix = d['prefix']
        self.client_name = d['config']
        self.client = dec.repo[S3Config.get_item_type(), self.client_name]
