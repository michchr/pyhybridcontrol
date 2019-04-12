import sys
import logging
import re
import pymongo as pym
from pymongo import MongoClient, errors as PymErrors, CursorType
from pymongo.collection import Collection, Cursor
from pymongo.database import Database
import pandas as pd
from datetime import datetime as DateTime
from dateutil.parser import parse as parse_datetime
from functools import reduce
from tools.grid_dataframe import MicroGridDataFrame

pd.set_option('mode.chained_assignment', 'raise')

console_hand = logging.StreamHandler(sys.stdout)
console_hand.setLevel(logging.INFO)
console_hand.setFormatter(logging.Formatter(fmt='\"%(pathname)s:%(lineno)s\"\n'
                                                '%(levelname)s:%(name)s:%(message)s'))

logger = logging.getLogger(__name__)
logger.propagate = False
if logger.hasHandlers(): logger.handlers = []
logger.addHandler(console_hand)
logger.setLevel(logging.INFO)

_DEV_TAG_MAP = {
    'dewh': 'GEY',
    'pm': 'PM',
    'inv': 'INV',
    'ccg': 'CCG'
}

_DEV_DEFAULT_FIELDS = {
    'dewh': ['Temp', 'Status'],
    'pm': ['ForActiveEnergy_Tot', 'RevActiveEnergy_Tot', 'ActivePower_Tot', 'ApparentPower_Tot']
}

_DEV_STRING_FIELDS = ['ID', 'SN', 'DevType']


# def deep_get(doc, key_list, default=None):
#     def reducer(d, key):
#         if isinstance(d, dict):
#             return d.get(key)
#         else:
#             return default
#
#     return reduce(reducer, key_list, doc)


class MongoInterface():

    def __init__(self, database=None, collection=None, host='localhost', port=27017, connectTimeoutMS=3000,
                 serverSelectionTimeoutMS=3000):

        if isinstance(collection, Collection):
            self._collection = collection
            self._database = collection.database
            self._client = self._database.client
        elif isinstance(database, Database):
            self._database = database
            self._client = self._database.client
            self._collection = self._database[collection]
        else:
            try:
                self._client = MongoClient(host, port, connect=True, connectTimeoutMS=connectTimeoutMS,
                                           serverSelectionTimeoutMS=serverSelectionTimeoutMS)
                # The ismaster command is cheap and does not require auth.
                self._client.admin.command('ismaster')
            except PymErrors.ConnectionFailure:
                self._client.close()
                raise ConnectionError('Could not connect to database - server not available')

            if database not in self._client.list_database_names():
                logger.info("Database: '%s', does not exist - creating new", database)
            self._database = self._client[database]

            if collection not in self._database.list_collection_names():
                logger.info("Collection: '%s', does not exist in Database: '%s' - creating new", collection,
                            self._database.name)
            self._collection = self._database[collection]

            logger.info("Created MongoInterface using Database: '%s', and Collection: '%s'",
                        self._database.name, self._collection.name)

    @property
    def client(self):
        return self._client

    @property
    def database(self):
        return self._database

    @property
    def collection(self):
        return self._collection

    @collection.setter
    def collection(self, collection: Collection):
        self._collection = collection

    def get_many_dev_cursor(self, device_type, dev_ids,
                            self_col: Collection = None,
                            start_datetime: DateTime = None, end_datetime: DateTime = None,
                            fields=None, to_numeric=True,
                            limit=0):

        if (self_col is not None) and (self_col != self._collection):
            self.collection = self_col
            logger.info("MongoInterface now using Collection: '{%s}'", self.collection.name)

        select_all_fields = False
        if fields == 'all':
            select_all_fields = True
            fields = self.get_many_dev_fields(device_type, dev_ids, self.collection)
        elif fields is None:
            fields = _DEV_DEFAULT_FIELDS[device_type] or self.get_many_dev_fields(device_type, dev_ids, self.collection)
        elif (not isinstance(fields, list)):
            raise ValueError("fields must be list of strings or 'all'")

        mongo_dev_tag = _DEV_TAG_MAP.get(device_type)

        if mongo_dev_tag:
            str_id_list = [mongo_dev_tag + str(dev_id) for dev_id in dev_ids]
        else:
            raise ValueError("Device type: '{}' does not exist or is not supported".format(device_type))

        field_keys_numeric = {str_id: [str_id + "." + field for field in fields if field not in _DEV_STRING_FIELDS] for
                              str_id in str_id_list}
        field_keys_string = {str_id: [str_id + "." + field for field in fields if field in _DEV_STRING_FIELDS] for
                             str_id in str_id_list}

        field_keys = {str_id: (field_keys_numeric[str_id] + field_keys_string[str_id]) for str_id in str_id_list}

        dev_selection = [{'$and': [{str_id: {'$exists': 1}},
                                   {'$or': [
                                       {field_key: {'$ne': ""}} for field_key in field_keys[str_id]
                                   ]}
                                   ]
                          } for str_id in str_id_list
                         ]

        query = {'$and': [{"TimeStamp": {'$gte': start_datetime, '$lte': end_datetime}},
                          {'$or': dev_selection}
                          ]
                 }

        projection = {"_id": 0, 'TimeStamp': 1}

        #
        if not to_numeric:
            if select_all_fields:
                projection.update({str_id: 1 for str_id in str_id_list})
            else:
                projection.update({field_key: 1 for str_id in str_id_list for field_key in field_keys[str_id]})

            dev_cursor = self.collection.find(query, projection, sort=[("TimeStamp", 1)], limit=limit)
        else:
            converter_numeric = {field_key: {'$cond':
                                                 {'if':
                                                      {'$eq':  # Test if string value does not contain a '.'
                                                           [{'$indexOfBytes': ['$' + field_key, "."]}, -1]
                                                       },
                                                  'then':  # If not '.' convert to int
                                                      {'$convert':
                                                           {'input': '$' + field_key,
                                                            'to': "int",
                                                            'onError': float('nan')}
                                                       },
                                                  'else':  # else to float/double
                                                      {'$convert':
                                                           {'input': '$' + field_key,
                                                            'to': "double",
                                                            'onError': float('nan')}
                                                       }
                                                  }
                                             } for str_id in str_id_list for field_key in field_keys_numeric[str_id]}

            converter_string = {field_key: {'$cond':
                                                {'if': {'$eq': ['$' + field_key, '']},
                                                 'then': None,
                                                 'else': '$' + field_key
                                                 }
                                            } for str_id in str_id_list for field_key in field_keys_string[str_id]}

            projection.update(converter_numeric)
            projection.update(converter_string)

            pipeline = [{'$match': query},
                        {'$sort': {"TimeStamp": 1}},
                        {'$project': projection},
                        ]

            if limit:
                pipeline.insert(2, {'$limit': limit})

            dev_cursor = self.collection.aggregate(pipeline)

        return dev_cursor

    def get_one_dev_cursor(self, device_type, dev_id,
                           self_col: Collection = None,
                           start_datetime: DateTime = None, end_datetime: DateTime = None,
                           fields=None, convert=True,
                           limit=0):

        return self.get_many_dev_cursor(device_type, [dev_id], self_col=self_col, start_datetime=start_datetime,
                                        end_datetime=end_datetime, limit=limit, fields=fields, to_numeric=convert)

    def get_many_dev_raw_dataframe(self, device_type, dev_ids, cursor=None,
                                   self_col: Collection = None,
                                   start_datetime: DateTime = None, end_datetime: DateTime = None,
                                   fields=None, convert=True,
                                   limit=0):

        force_to_numeric = False
        if cursor is None:
            cursor = self.get_many_dev_cursor(device_type, dev_ids, self_col=self_col, start_datetime=start_datetime,
                                              end_datetime=end_datetime, limit=limit, fields=fields, to_numeric=convert)
        elif convert:
            force_to_numeric = True

        with cursor as cursor:
            list_cur = list(cursor)

        if len(list_cur) == 0:
            raw_df = pd.DataFrame()
        else:
            raw_df = pd.DataFrame(list_cur).set_index('TimeStamp').unstack().dropna()
            raw_df = pd.DataFrame.from_records(raw_df.values, index=raw_df.index)
            raw_df = raw_df[~raw_df.index.duplicated(keep='last')]
            raw_df = raw_df.unstack(0).reorder_levels([1, 0], axis=1).sort_index(axis=1)

            level_0 = [int(re.findall("[0-9]*[0-9]", device)[0]) for device in raw_df.columns.levels[0]]
            raw_df.columns.set_levels(level_0, level=0, inplace=True)
            raw_df.columns.names = [device_type.upper() + "_id", 'Tag']

            if force_to_numeric:
                idx = pd.IndexSlice
                numeric_fields = [i for i in raw_df.columns.levels[1] if i not in _DEV_STRING_FIELDS]
                string_fields = [i for i in raw_df.columns.levels[1] if i in _DEV_STRING_FIELDS]
                raw_df.loc[:, idx[:, numeric_fields]] = raw_df.loc[:, idx[:, numeric_fields]].apply(pd.to_numeric,
                                                                                                    errors='coerce')
                raw_df.loc[:, idx[:, string_fields]] = raw_df.loc[:, idx[:, string_fields]].replace('', None)

        return MicroGridDataFrame(raw_df, device_type=device_type)

    def get_one_dev_raw_dataframe(self, device_type, dev_id, cursor=None,
                                  self_col: Collection = None,
                                  start_datetime: DateTime = None, end_datetime: DateTime = None,
                                  fields=None, convert=True,
                                  limit=0):

        return self.get_many_dev_raw_dataframe(device_type, [dev_id], self_col=self_col, start_datetime=start_datetime,
                                               end_datetime=end_datetime, limit=limit, fields=fields, cursor=cursor,
                                               convert=convert)

    def get_many_dev_fields(self, device_type, dev_ids, self_col: Collection = None):
        if (self_col is not None) and (self_col != self._collection):
            self.collection = self_col
            logger.info("MongoInterface now using Collection: '{%s}'", self.collection.name)

        mongo_dev_tag = _DEV_TAG_MAP.get(device_type)
        str_id_list = [mongo_dev_tag + str(dev_id) for dev_id in dev_ids]
        query = {}
        sort = [('TimeStamp', -1)]
        proj = {'_id': 0}
        proj.update({str_id: 1 for str_id in str_id_list})

        doc = self.collection.find_one(query, proj, sort=sort, max_time_ms=100)

        fields = set()
        for key, value in doc.items():
            fields.update(list(value.keys()))

        return list(fields)

    def get_one_dev_fields(self, device_type, dev_id, self_col: Collection = None):
        return self.get_many_dev_fields(device_type, [dev_id], self_col=self_col)

    def get_many_dev_ids(self, device_types=None):
        query = {}
        sort = [('TimeStamp', -1)]
        proj = {'_id': 0, 'TimeStamp': 0, 'file': 0}
        doc = self.collection.find_one(query, proj, sort=sort, max_time_ms=100)
        devices = list(doc.keys())

        rev_dev_tag_map = {value: key for key, value in _DEV_TAG_MAP.items()}

        reg_alpha = re.compile("[a-zA-Z]+")
        reg_num = re.compile("[0-9]*[0-9]")
        device_ids = {}
        for device in devices:
            try:
                device_type = rev_dev_tag_map.get(reg_alpha.match(device)[0])
            except AttributeError:
                device_type = None
            if device_type is not None:
                dev_id = int(reg_num.search(device)[0])
                if device_type in device_ids.keys():
                    device_ids[device_type].add(dev_id)
                else:
                    device_ids[device_type] = {dev_id}

        if device_types != 'all' or device_types is not None:
            device_ids = {device_type: device_ids.get(device_type) for device_type in device_types}

        return device_ids

    def get_one_dev_ids(self, device_type):
        return self.get_many_dev_ids(device_types=[device_type]).get(device_type)

    def set_string_field_to_datetime(self, self_col: Collection = None, field_tag='TimeStamp'):

        if (self_col is not None) and (self_col != self._collection):
            self.collection = self_col
            logger.info("MongoInterface now using Collection: '{%s}'", self.collection.name)

        # find docs with field tags that are not of datetime format
        query = {field_tag: {'$not': {'$type': "date"}}}
        proj = {field_tag: 1}

        with self.collection.find(query, proj) as cursor:
            bulk_requests = []
            for doc in cursor:
                timestamp = parse_datetime(doc[field_tag])
                bulk_requests.append(pym.UpdateOne({
                    '_id': doc['_id']
                }, {
                    '$set': {
                        field_tag: timestamp
                    }
                }))

        if bulk_requests:
            result = self.collection.bulk_write(bulk_requests).modified_count
        else:
            result = 0

        logger.info("Modified TimeStamp type for %s documents", result)
        return result

    def append_from_collection(self, from_col: Collection, self_col: Collection = None, limit=0, newest=True,
                               override=False):

        if (self_col is not None) and (self_col != self._collection):
            self.collection = self_col
            logger.info("MongoInterface now using Collection: '{%s}'", self.collection.name)

        if from_col.count_documents({}) == 0:
            raise ValueError('Source _collection is empty, nothing to copy')
        if not override and from_col == self_col:
            raise ValueError('from_col == self_col, cannot append to self from self, unless override is set')

        order = pym.DESCENDING if newest else pym.ASCENDING
        query = {}
        proj = {'_id': 0}
        sort = [('_id', order)]

        with from_col.find(query, proj, sort=sort, limit=limit) as cursor:
            requests = []
            for doc in cursor:
                requests.append(pym.InsertOne(doc))

        try:
            result = self.collection.bulk_write(requests)
        except PymErrors.BulkWriteError as bwe:
            raise IndexError(
                "BulkWriteError with errmsg: {}".format(bwe.details['writeErrors'][0]['errmsg'])) from None

        num_appended = result.inserted_count
        logger.info("Appended %s documents", num_appended)
        return num_appended

    def clone_from_collection(self, from_col: Collection, self_col: Collection = None, limit=0, newest=True,
                              override=False):

        if (self_col is not None) and (self_col != self._collection):
            self.collection = self_col
            logger.info("MongoInterface now using Collection: '{%s}'", self.collection.name)

        order = pym.DESCENDING if newest else pym.ASCENDING

        if limit:
            pipeline = [{'$sort': {'_id': order}},
                        {'$limit': limit},
                        {'$project': {'_id': 0}},
                        {'$out': self.collection.name}
                        ]
        else:
            pipeline = [{'$sort': {'_id': order}},
                        {'$project': {'_id': 0}},
                        {'$out': self.collection.name}
                        ]

        if not override and from_col == self.collection:
            raise ValueError('from_col == self_col, cannot clone self to self, unless override is set')
        elif not override and self.collection.count_documents({}) > 0:
            raise ValueError("Output database already contains items, set override to force")
        else:
            from_col.database.command('aggregate', from_col.name, pipeline=pipeline, explain=False)

        num_cloned = self.collection.count_documents({})
        logger.info("Cloned %s documents", num_cloned)
        return num_cloned

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        self.client.close()


if __name__ == '__main__':
    import timeit
    from timeit import default_timer as Timer

    mi = MongoInterface(database='site_data', collection='Kikuyu')
    st = Timer()
    mi.set_string_field_to_datetime()
    mi.collection.create_index("TimeStamp")
    print("Time taken is {}".format(Timer() - st))

    #
    # start_date = DateTime(2018, 9, 30, 12, 0)
    # end_date = DateTime(2018, 10, 30, 12, 3)
    #
    # raw_df = mi.get_many_dev_raw_dataframe('pm', [0], start_datetime=start_date, end_datetime=end_date,
    #                                        fields='all', convert=True)
    #

    # raw_df
    #
    # st = Timer()
    # df = raw_df.apply(pd.to_numeric, errors='coerce')
    # print("Time taken is {}".format(Timer() - st))
    #
    # print(df.info())

    # raw_df = mi.get_one_dev_cursor('dewh', 5, start_datetime=start_date, end_datetime=end_date)
    # raw_df = list(raw_df)
    # raw_df = pd.DataFrame(raw_df).set_index('TimeStamp')
    # raw_df = pd.DataFrame(raw_df.iloc[:,0].values.tolist(),index=raw_df.index, columns=['Temp', 'Status',
    # 'Error'])
    #
    # df2 = mi.get_one_dev_cursor('dewh', 6, start_datetime=start_date, end_datetime=end_date)
    # df2 = list(df2)
    # df2 = pd.DataFrame(df2).set_index('TimeStamp')
    # df2 = pd.DataFrame(df2.iloc[:, 0].values.tolist(), index=df2.index, columns=['Temp', 'Status', 'Error'])
    #
    # raw_df = mi.get_one_dev_cursor('pmo', 0, start_datetime=start_date, end_datetime=end_date)
    # raw_df = list(raw_df)
    # raw_df = pd.DataFrame(raw_df).set_index('TimeStamp')
    #
    #
    # raw_df = pd.DataFrame(raw_df.iloc[:, 0].values.tolist(), index=raw_df.index, columns=[
    # 'ForActiveEnergy_Tot']).astype(float)
    # print(raw_df)
    # raw_df.plot()

    # print(raw_df)

    # raw_df = raw_df.reset_index().drop_duplicates(subset='TimeStamp').set_index('TimeStamp')
    #
    # df1 = raw_df.resample('1min', closed='right').ffill(1)
    # df2 = raw_df.resample('1min', closed='left').ffill(1)
    # raw_df = raw_df.resample('1min').ffill(1)
    #
    # df1['isNaN'] = np.isnan(df1.Temp.values).astype(np.int)
    # df1.isNaN = (df1.isNaN.rolling(15).mean() >= 1).astype(np.int)
    # df1['Temp2'] = df1.Temp + 50
    # df1['Status2'] = df1.Status * 10 - 20
    # df1.Temp = df1.Temp.interpolate()
    # df1.Status = df1.Status.ffill()
    #
    # df2['isNaN'] = np.isnan(df1.Temp.values).astype(np.int)
    # df2.isNaN = (df2.isNaN.rolling(15).mean() >= 1).astype(np.int)
    # df2.Temp = df2.Temp.interpolate()
    # df2.Status = df2.Status.ffill()
    #
    # # df1 = df1.resample('15Min', closed='right').mean()
    # df2 = df2.resample('15Min').mean()
    #
    # import matplotlib.pyplot as plt
    #
    # plt.style.use('default')
    #
    # df1.Status *= 10
    #
    # df1.isNaN -= 2
    #
    # # plt.figure()
    # # raw_df.Temp.plot()
    # # plt.figure()
    # fig, ax = plt.subplots(1, 1)
    # df1.plot(drawstyle="steps-post", ax=ax)
    # # df2.plot(drawstyle="steps-post", ax=ax)
    # # plt.figure()
    # # df1.isNaN.plot(drawstyle="steps-post")
    #
    # print(df1.describe())
    #
    # plt.show()
