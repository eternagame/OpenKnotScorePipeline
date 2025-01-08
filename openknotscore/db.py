'''
Custom file format for storing pipeline data, optimized for random access (without fully
loading into memory) and data deduplication
'''

import struct
import pickle
from typing import Any
from collections import OrderedDict

DB_VERSION = 1

class Collection:
    def __init__(self):
        self._values = OrderedDict[bytes, int]()

    def insert(self, value: Any):
        pkl = pickle.dumps(value)
        if len(pkl) > 2**64:
            raise ValueError('DB value must be less than 2^64 bytes')
        if len(self._values) >= 2**64:
            raise ValueError('DB collection must contain less than 2^64 items')
        item_id = self._values.get(pkl)
        if item_id is None:
            item_id = len(self._values)
            self._values[pkl] = item_id
        return item_id
    
    def serialize(self):
        index = b''
        data = b''
        for value in self._values.keys():
            index += struct.pack('<Q', len(data))
            data += value

        return struct.pack('<Q', len(self._values)) + struct.pack('<Q', len(data)) + index + data
        

class DBWriter:
    def __init__(self, path: str):
        self._f = open(path, 'wb')
        self._collections: dict[str, Collection] = {}
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, exc_tb):
        self.write()
        self.close()
    
    def write(self):
        index = b''
        data = b''
        for (name, collection) in self._collections.items():
            collection_bytes = collection.serialize()
            name_bytes = name.encode()
            index += struct.pack('<B', len(name_bytes))
            index += name_bytes
            index += struct.pack('<Q', len(collection_bytes))
            data += collection_bytes
        
        self._f.write(b'OKSPDB')
        # Version
        self._f.write(struct.pack('<B', DB_VERSION))
        self._f.write(struct.pack('<B', len(self._collections)))
        self._f.write(index)
        self._f.write(data)

    def close(self):
        self._f.close()

    def create_collection(self, name: str):
        if name in self._collections:
            raise RuntimeError(f'Collection {name} already exists')
        if len(name.encode()) > 255:
            raise ValueError('Collection name must be less than 256 bytes')
        if len(self._collections) > 255:
            raise ValueError('DB must contain less than 256 collections')
        self._collections[name] = Collection()
        return self._collections[name]

class DBReader:
    def __init__(self, path: str):
        self._f = open(path, 'rb')

        if self._f.read(6) != b'OKSPDB':
            raise ValueError(f'File {path} is not a valid OKSP database')
        self.version = struct.unpack('<B', self._f.read(1))[0]
        if self.version > DB_VERSION:
            raise ValueError(f'OKSP database at {path} uses DB version {self.version}, but only up to {DB_VERSION} is supported')
        
        collection_count = struct.unpack('<B', self._f.read(1))[0]
        collection_offset = 0
        self._collections = {}
        for _ in range(collection_count):
            name_length = struct.unpack('<B', self._f.read(1))[0]
            name = self._f.read(name_length).decode()
            data_length = struct.unpack('<Q', self._f.read(8))[0]
            self._collections[name] = {'offset': collection_offset}
            collection_offset += data_length

        header_size = self._f.tell()
        for name in self._collections:
            self._collections[name]['offset'] += header_size
            self._f.seek(self._collections[name]['offset'])
            self._collections[name]['entries'] = struct.unpack('<Q', self._f.read(8))[0]
            self._collections[name]['data_length'] = struct.unpack('<Q', self._f.read(8))[0]
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, exc_tb):
        self.close()

    def close(self):
        self._f.close()

    def get(self, collection: str, index: int):
        collection = self._collections[collection]
        self._f.seek(collection['offset'] + 16 + 8*index)
        offset = struct.unpack('<Q', self._f.read(8))[0]
        if index < collection['entries'] - 1:
            length = struct.unpack('<Q', self._f.read(8))[0] - offset
        else:
            length = collection['data_length'] - offset
        
        self._f.seek(collection['offset'] + 16 + collection['entries']*8 + offset)
        return pickle.loads(self._f.read(length))
