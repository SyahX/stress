import pymongo
import logging
logger = logging.getLogger()

class User(object):
    def __init__(self, db):
        self.db = db

        # self._get_all_users()
        self._get_all_labels()
        self._clean_no_label()

    def _build_user_config(self):
        config = {}
        config['_id'] = 0
        config['id'] = 1
        config['name'] = 1
        config['location'] = 1
        config['friends_count'] = 1
        config['followers_count'] = 1
        config['statuses_count'] = 1
        self.config = config

    def _build_blog_config(self):
        config = {}
        config['_id'] = 0
        config['id'] = 1
        config['user_id'] = 1
        config['text'] = 1
        self.config = config

    def _get_all_users(self):
        self._build_user_config()

        total = 0
        self.info = {}
        logger.info("total users : %d" % self.db.test.users.find().count())
        for user in self.db.test.users.find({}, self.config):
            total += 1
            self.info[user['id']] = user
            self.info[user['id']]['label'] = 'no'
            if total % 100000 == 0:
                logger.info("\tprocessing %d ..." % total)
        logger.info("load total : %d" % len(self.info))

    def _get_all_labels(self):
        self._build_blog_config()

        total = 0
        cnt = 0
        logger.info("total blogs : %d" % self.db.test.microblogs.find().count())
        for blog in self.db.test.microblogs.find({}, self.config):
            total += 1
            if '压力大' in blog['text']:
                cnt += 1
                print (blog)
            if total % 100000 == 0:
                logger.info("\tprocessing %d (%d)..." % (total, cnt))


    def _clean_no_label(self):
        info = self.info
        self.info = {}
        for idx, user in info.items():
            if (user['label'] != 'no'):
                self.info[idx] = user
        logger.info("after clean : %d" % (len(self.info)))

class Data(object):
    def __init__(self, config):
        self.db = pymongo.MongoClient("mongodb://166.111.139.44:27017/")
        self.config = config
        self.user = User(self.db)

    def get_feature(self):
        return
