"""
UnitTest Module

....................................................................................................
MIT License
Copyright (c) 2021-2023 AUT Iran, Mohammad H Forouhesh
Copyright (c) 2021-2022 MetoData.ai, Mohammad H Forouhesh
....................................................................................................
This module serves as unit testing for various functionalities in the code.
"""

import unittest

from tracking_policy_agendas.api import downloader
from tracking_policy_agendas.classifiers.pa_clf import PAClf
from tracking_policy_agendas.classifiers.xgb_clf import XgbClf
from tracking_policy_agendas.classifiers.naive_bayes_clf import GNBClf
from tracking_policy_agendas.classifiers.lasso_clf import LassoClf


class XgbTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.clf = XgbClf(load_path='xgb_vaccine')

    def test_xgb_api(self) -> None:
        self.assertRaises(Exception, downloader, path='wrong-path')

    def test_xgb_soundness(self) -> None:
        self.assertNotEqual(self.clf['تزریق واکسن هم اجباری شد'], self.clf['کرونا با ماسک و واکسن هم از بین نمیرود'])
        self.assertNotEqual(self.clf['واکسیناسیون عمومی کزاز در ریشه کنی این بیماری بسیار مثمر ثمر بوده است'],
                            self.clf['تزریق دوز سوم واکسن کرونا هم تصویب شد'])

    def test_xgb_completeness(self) -> None:
        self.assertEqual(self.clf.vec_predict('دوز سوم واکسن کرونا'), 1)
        self.assertEqual(self.clf['رئيس‌جمهور جمهوری اسلامی'], 0)
        self.assertEqual(self.clf['بورس نماد اقتصاد بحران زده‌ی ایران'], 0)


class PATestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.clf = PAClf(load_path='pa_vaccine')

    def test_pa_api(self) -> None:
        self.assertRaises(Exception, downloader, path='wrong-path')

    def test_pa_soundness(self) -> None:
        self.assertNotEqual(self.clf['تزریق واکسن هم اجباری شد'],
                         self.clf['کرونا با ماسک و واکسن هم از بین نمیرود'])
        self.assertNotEqual(self.clf['واکسیناسیون عمومی کزاز در ریشه کنی این بیماری بسیار مثمر ثمر بوده است'],
                            self.clf['تزریق دوز سوم واکسن کرونا هم تصویب شد'])

    def test_pa_completeness(self) -> None:
        self.assertEqual(self.clf.vec_predict('دوز سوم واکسن کرونا'), 1)
        self.assertEqual(self.clf['بورس نماد اقتصاد بحران زده‌ی ایران'], 0)


class LassoTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.clf = LassoClf(load_path='lasso_vaccine')

    def test_lasso_api(self) -> None:
        self.assertRaises(Exception, downloader, path='wrong-path')

    def test_lasso_soundness(self) -> None:
        self.assertNotEqual(self.clf['تزریق دوز سوم واکسن هم تصویب شد'],
                         self.clf['کرونا با ماسک و واکسن هم از بین نمیرود'])
        self.assertNotEqual(self.clf['واکسیناسیون عمومی کزاز در ریشه کنی این بیماری بسیار مثمر ثمر بوده است'],
                            self.clf['تزریق دوز سوم واکسن کرونا هم تصویب شد'])

    def test_lasso_completeness(self) -> None:
        self.assertEqual(self.clf.vec_predict('دوز سوم واکسن کرونا'), 1)


class GNBTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.clf = GNBClf(load_path='gnb_vaccine')

    def test_gnb_api(self) -> None:
        self.assertRaises(Exception, downloader, path='wrong-path')

    def test_gnb_soundness(self) -> None:
        self.assertEqual(self.clf['تزریق دوز سوم واکسن کرونا هم تصویب شد'],
                         self.clf['کرونا با ماسک و واکسن هم از بین نمیرود'])

    def test_gnb_completeness(self) -> None:
        self.assertEqual(self.clf.vec_predict('دوز سوم واکسن کرونا'), 1)
        self.assertEqual(self.clf['رئيس‌جمهور جمهوری اسلامی'], 0)
        self.assertEqual(self.clf['بورس نماد اقتصاد بحران زده‌ی ایران'], 0)


if __name__ == '__main__':
    unittest.main()
