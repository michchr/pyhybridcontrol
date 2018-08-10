import sys
import logging
import pymongo as pym
from pymongo import MongoClient, errors as PymErrors, CursorType
from pymongo.collection import Collection, Cursor
from pymongo.database import Database
import pandas as pd
import numpy as np
from datetime import datetime as DateTime
from dateutil.parser import parse as parse_datetime
from functools import reduce

pd.set_option('mode.chained_assignment', 'raise')
# db.authenticate(MONGO_USER, MONGO_PASS)

console_hand = logging.StreamHandler(sys.stdout)
console_hand.setLevel(logging.INFO)
console_hand.setFormatter(logging.Formatter(fmt='%(levelname)s:%(name)s:%(message)s'))

logger = logging.getLogger(__name__)
logger.propagate = False
if logger.hasHandlers(): logger.handlers = []
logger.addHandler(console_hand)
logger.setLevel(logging.INFO)

_DEV_TAG_MAP = {
    'dewh': 'GEY',
    'pm': 'PM',
    'inv': 'INV',
}

_DEV_DEFAULT_FIELDS = {
    'dewh': ['Temp', 'Status'],
    'pm': ['ForActiveEnergy_Tot', 'RevActiveEnergy_Tot', 'ActivePower_Tot', 'ApparentPower_Tot']
}


def deep_get(doc, key_list, default=None):
    def reducer(d, key):
        if isinstance(d, dict):
            return d.get(key)
        else:
            return default

    return reduce(reducer, key_list, doc)


class MongoInterface():

    def __init__(self, database=None, collection=None, host='localhost', port=27017):

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
                self._client = MongoClient(host, port)
            except:
                raise ConnectionError('Could not connect to database')

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

    def get_many_dev_cursor(self, device_type, dev_ids, self_col: Collection = None, start_time: DateTime = None,
                            end_time: DateTime = None, limit=0, fields=None):

        if (self_col is not None) and (self_col != self._collection):
            self.collection = self_col
            logger.info("MongoInterface now using Collection: '{%s}'", self.collection.name)

        if fields == 'all':
            select_all_fields = True
            fields = None
        else:
            select_all_fields = False

        if (not isinstance(fields, list)) and (fields is not None):
            raise ValueError("fields must be list of strings or 'all'")

        mongo_dev_tag = _DEV_TAG_MAP.get(device_type)
        mongo_dev_fields = _DEV_DEFAULT_FIELDS.get(device_type)

        if mongo_dev_tag:
            str_id_list = [mongo_dev_tag + str(dev_id) for dev_id in dev_ids]
            if fields is None:
                fields = mongo_dev_fields
        else:
            raise ValueError("Device type: '{}' does not exist or is not supported".format(device_type))

        dev_selection = [{'$and': [{str_id: {'$exists': 1}},
                                   {'$or': [
                                       {str_id + "." + field: {'$ne': ""}} for field in fields
                                   ]}
                                   ]
                          } for str_id in str_id_list
                         ]

        query = {'$and': [{"TimeStamp": {'$gte': start_time, '$lte': end_time}},
                          {'$or': dev_selection}
                          ]
                 }

        projection = {"_id": 0, 'TimeStamp': 1}
        if select_all_fields:
            projection.update({str_id: 1 for str_id in str_id_list})
        else:
            projection.update({str_id + "." + field: 1 for field in fields for str_id in str_id_list})

        cursor = self.collection.find(query, projection, sort=[("TimeStamp", 1)], limit=limit,
                                      cursor_type=CursorType.NON_TAILABLE)

        return cursor

    def get_one_dev_cursor(self, device_type, dev_id, self_col: Collection = None, start_time: DateTime = None,
                           end_time: DateTime = None, limit=0, fields=None):

        return self.get_many_dev_cursor(device_type, [dev_id], self_col=self_col, start_time=start_time,
                                        end_time=end_time, limit=limit, fields=fields)

    def get_many_dev_raw_dataframe(self, device_type, dev_ids, self_col: Collection = None, start_time: DateTime = None,
                                   end_time: DateTime = None, limit=0, fields=None, cursor=None):

        if cursor is None:
            cursor = self.get_many_dev_cursor(device_type, dev_ids, self_col=self_col, start_time=start_time,
                                              end_time=end_time, limit=limit, fields=fields)
        with cursor as cursor:
            list_cur = list(cursor)

        if len(list_cur) == 0:
            raw_df = pd.DataFrame()
        else:
            raw_df = pd.DataFrame(list_cur).set_index('TimeStamp').unstack().dropna()
            raw_df = pd.DataFrame(raw_df.values.tolist(), index=raw_df.index)
            raw_df = raw_df[~raw_df.index.duplicated(keep='last')]
            raw_df = raw_df.unstack(0).reorder_levels([1, 0], axis=1).sort_index(axis=1)

        return raw_df

    def get_one_dev_raw_dataframe(self, device_type, dev_id, self_col: Collection = None, start_time: DateTime = None,
                                  end_time: DateTime = None, limit=0, fields=None, cursor=None):

        return self.get_many_dev_raw_dataframe(device_type, [dev_id], self_col=self_col, start_time=start_time,
                                               end_time=end_time, limit=limit, fields=fields, cursor=cursor)

    def get_dev_tags(self, device_type, dev_id, self_col: Collection = None):
        if (self_col is not None) and (self_col != self._collection):
            self.collection = self_col
            logger.info("MongoInterface now using Collection: '{%s}'", self.collection.name)





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

    # todo not completed
    def set_string_field_to_float(self, self_col: Collection = None, field_tag='GEY1.Temp'):

        if (self_col is not None) and (self_col != self._collection):
            self.collection = self_col
            logger.info("MongoInterface now using Collection: '{%s}'", self.collection.name)

        # find docs with field tags that are not of a certain type
        query = {field_tag: {'$not': {'$type': "double"}}}
        proj = {field_tag: 1}

        with self.collection.find(query, proj) as cursor:
            bulk_requests = []
            for doc in cursor:
                try:
                    value = float(deep_get(doc, field_tag.split('.')))
                except ValueError:
                    value = float('nan')
                except TypeError:
                    value = float('nan')

                bulk_requests.append(pym.UpdateOne({
                    '_id': doc['_id']
                }, {
                    '$set': {
                        field_tag: value
                    }
                }))

        if bulk_requests:
            result = self.collection.bulk_write(bulk_requests).modified_count
        else:
            result = 0

        logger.info("Modified field type for %s documents", result)
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
            raise IndexError("BulkWriteError with errmsg: {}".format(bwe.details['writeErrors'][0]['errmsg'])) from None

        num_appended = result.inserted_count
        logger.info("Appended %s documents", num_appended)
        return num_appended

    def clone_from_collection(self, from_col: Collection, self_col: Collection = None, limit=0, newest=True,
                              override=False):

        if (self_col is not None) and (self_col != self._collection):
            self.collection = self_col
            logger.info("MongoInterface now using Collection: '{%s}'", self.collection.name)

        order = pym.DESCENDING if newest else pym.ASCENDING
        if limit is None:
            pipeline = [{'$sort': {'_id': order}},
                        {'$project': {'_id': 0}},
                        {'$out': self.collection.name}
                        ]
        else:
            pipeline = [{'$sort': {'_id': order}},
                        {'$limit': limit},
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

    def close(self):
        self.client.close()


if __name__ == '__main__':
    import timeit
    from timeit import default_timer as Timer

    mi = MongoInterface(database='site_data', collection='Kikuyu')
    # mi.clone_collection(kik)
    mi.set_string_field_to_datetime()
    mi.collection.create_index("TimeStamp")

    start_date = DateTime(2018, 7, 28, 12, 0)
    end_date = DateTime(2018, 7, 30, 12, 3)

    st = Timer()
    raw_df = mi.get_many_dev_raw_dataframe('pm', [0], start_time=start_date, end_time=end_date, fields='all')

    print("Time taken is {}".format(Timer() - st))

    st = Timer()
    df = raw_df.apply(pd.to_numeric, errors='coerce')
    print("Time taken is {}".format(Timer() - st))
    print(raw_df.head(2))

    # raw_df = mi.get_one_dev_cursor('dewh', 5, start_time=start_date, end_time=end_date)
    # raw_df = list(raw_df)
    # raw_df = pd.DataFrame(raw_df).set_index('TimeStamp')
    # raw_df = pd.DataFrame(raw_df.iloc[:,0].values.tolist(),index=raw_df.index, columns=['Temp', 'Status', 'Error'])
    #
    # df2 = mi.get_one_dev_cursor('dewh', 6, start_time=start_date, end_time=end_date)
    # df2 = list(df2)
    # df2 = pd.DataFrame(df2).set_index('TimeStamp')
    # df2 = pd.DataFrame(df2.iloc[:, 0].values.tolist(), index=df2.index, columns=['Temp', 'Status', 'Error'])
    #
    # raw_df = mi.get_one_dev_cursor('pmo', 0, start_time=start_date, end_time=end_date)
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
