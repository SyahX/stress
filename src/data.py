import os
import json
import pymongo
import logging
import numpy as np
from pyltp import Segmentor
import utils as Utils
logger = logging.getLogger()

class User(object):
    def __init__(self, db, dim):
        self.db = db
        self.dim = dim

        self._get_all_users()
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

    def _build_blog_config(self, tags):
        config = {}
        config['_id'] = 0
        config['id'] = 1
        config['user_id'] = 1
        config['text'] = 1
        self.config = config
        # self.query = {"text" : {'$regex': '.*[' + ','.join(tags) + '].*'}}
        self.query = {}
        logger.info(self.query)


    def _get_all_users(self):
        self._build_user_config()

        total = 0
        self.info = {}
        logger.info("total users : %d" % self.db.test.users.find().count())
        for user in self.db.test.users.find({}, self.config):
            total += 1
            if total % 100000 == 0:
                logger.info("\tprocessing %d ..." % total)

            self.info[user['id']] = user
            self.info[user['id']]['label'] = 'no'
        logger.info("load total : %d" % len(self.info))

    def _get_all_labels(self):
        true_tags = ['烦躁', '烦死了', '烦死人了', '压力大', '加班', '焦虑']
        false_tags = ['高兴', '开心'] #, '好幸福', '完美', '高兴', '快乐']
        self._build_blog_config(true_tags + false_tags)

        total = 0
        cntx = 0
        cnty = 0
        logger.info("total blogs : %d" % self.db.test.microblogs.find(self.query).count())
        for blog in self.db.test.microblogs.find(self.query, self.config):
            total += 1
            if total % 10000000 == 0:
                logger.info("\tprocessing %d (%d, %d)..." % (total, cntx, cnty))

            text = blog['text']
            if 'user_id' not in blog:
                continue
            if blog['user_id'] not in self.info:
                continue
            if self.info[blog['user_id']]['label'] == 'not':
                continue

            tags = ['@', '【', '『', '#', 'http']
            flag = False
            for tag in tags:
                if tag in text:
                    flag = True
                    continue
            if flag:
                continue

            tags = true_tags
            for tag in tags:
                if tag in text:
                    if self.info[blog['user_id']]['label'] == 'false':
                        self.info[blog['user_id']]['label'] = 'not'
                        cnty -= 1
                        break
                    if self.info[blog['user_id']]['label'] == 'no':
                        self.info[blog['user_id']]['label'] = 'true'
                        cntx += 1
                    # logger.info(text)
                    break

            tags = false_tags
            for tag in tags:
                if tag in text:
                    if self.info[blog['user_id']]['label'] == 'true':
                        self.info[blog['user_id']]['label'] = 'not'
                        cntx -= 1
                        break
                    if self.info[blog['user_id']]['label'] == 'no':
                        self.info[blog['user_id']]['label'] = 'false'
                        cnty += 1
                    break


    def _clean_no_label(self):
        info = self.info
        self.info = {}
        cntx = cnty = 0
        for idx, user in info.items():
            if user['label'] == 'true':
                self.info[idx] = user
                self.info[idx]['input'] = np.zeros(self.dim)
                cntx += 1
        for idx, user in info.items():
            if user['label'] == 'false':
                self.info[idx] = user
                self.info[idx]['input'] = np.zeros(self.dim)
                cnty += 1
                if (cnty == cntx):
                    break
        logger.info("after clean : %d (%d / %d)" % (len(self.info), cntx, cnty))


    def info(self):
        return self.info


class Data(object):
    def __init__(self, config):
        self.db = pymongo.MongoClient("mongodb://166.111.139.44:27017/")
        self.config = config
        self.dict = Utils.load_dict(config['dict'])
        self.pop_dict = Utils.load_pop_dict(config['pop_dict'])

        if os.path.exists('user.json'):
            self.user = json.load(open('user.json', 'r'))
        else:
            self.user = User(self.db, self.config['input_size'])
            self.user = self.user.info
            self._get_feature()

            json.dump(self.user, open('user.json', 'w'))

        self._get_data()
        if config['mini_data']:
            self.pos_data = self.pos_data[:100]
            self.neg_data = self.neg_data[:100]

        length = len(self.pos_data)
        logger.info("total data : %d" % (length * 2))
        """
        ptr = length // 5 * 4
        self.train = self.pos_data[:ptr] + self.neg_data[:ptr]
        self.test = self.pos_data[ptr:] + self.neg_data[ptr:]
        """
        ptr = length // 5 * 4
        self.train = self.pos_data[:ptr // 2] + self.neg_data[:ptr]
        self.test = self.pos_data[ptr + (length - ptr) // 2:] + self.neg_data[ptr:]
        logger.info("train size : %d" % len(self.train))
        logger.info(" test size : %d" % len(self.test))

    def _get_feature(self):
        config = {'id': 1, 'user_id': 1, 'text': 1}
        segmentor = Segmentor()
        segmentor.load(os.path.join(self.config['ltp_model'], 'cws.model'))

        self.input = []
        cnt = 0
        for blog in self.db.test.microblogs.find({}, config):
            cnt += 1
            if cnt % 10000000 == 0:
                logger.info("\tprocessing %d ..." % cnt)

            if 'user_id' not in blog:
                continue

            idx = blog['user_id']
            if idx not in self.user:
                continue
            words = list(segmentor.segment(blog['text']))
            score = []
            for word in words:
                if word in self.dict:
                    ptr = self.dict[word][0] * 5 + self.dict[word][1] // 2
                    self.user[idx]['input'][ptr] += 1
                if word == ',' or word == '，':
                    self.user[idx]['input'][105] += 1
                if word == '.' or word == '。':
                    self.user[idx]['input'][106] += 1
                if word == '?' or word == '？':
                    self.user[idx]['input'][107] += 1
                if word == '@':
                    self.user[idx]['input'][108] += 1
                if word in self.pop_dict:
                    score.append(self.pop_dict[word])
            if (len(score) > 0):
                self.user[idx]['input'][110] += sum(score) / len(score)
        segmentor.release()

        for idx, info in self.user.items():
            info['input'] = info['input'].tolist()

    def _get_data(self):
        self.pos_data = []
        self.neg_data = []
        for idx, info in self.user.items():
            if info['label'] == 'true':
                self.pos_data.append((info['input'], 1))
            else:
                self.neg_data.append((info['input'], 0))

