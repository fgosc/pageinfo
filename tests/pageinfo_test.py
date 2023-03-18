import os
import re
import unittest
from logging import getLogger
from pathlib import Path

import cv2
import pytesseract

import pageinfo

logger = getLogger(__name__)
here = os.path.dirname(os.path.abspath(__file__))


def get_images_absdir(dirname):
    return os.path.join(here, 'images', dirname)


class PageinfoTest(unittest.TestCase):
    def _test_guess_pageinfo(self, images_dir, expected):
        for entry in Path(images_dir).glob("**/*"):
            if entry.is_dir():
                continue
            if entry.suffix not in ('.png', '.jpg'):
                logger.warning("not a image file: %s", entry)
                continue
            impath = str(entry)
            relpath = str(entry.relative_to(images_dir))
            if relpath not in expected:
                continue

            with self.subTest(image=impath):
                im = cv2.imread(impath)
                try:
                    actual = pageinfo.guess_pageinfo(im)
                    self.assertEqual(actual, expected[relpath])

                except Exception as e:
                    self.fail(f'{impath}: {e}')

    def _test_detect_qp_region(self, images_dir, expected):
        for entry in Path(images_dir).glob("**/*"):
            if entry.is_dir():
                continue
            if entry.suffix not in ('.png', '.jpg'):
                logger.warning("not a image file: %s", entry)
                continue
            impath = str(entry)
            relpath = str(entry.relative_to(images_dir))
            if relpath not in expected:
                continue

            with self.subTest(image=impath):
                im = cv2.imread(impath)
                try:
                    coordinates = pageinfo.detect_qp_region(im)
                    _expected = expected[relpath]
                    if _expected is None:
                        self.assertIsNone(coordinates)
                        continue

                    topleft, bottomright = coordinates
                    qp_region = im[topleft[1]:bottomright[1], topleft[0]:bottomright[0]]
                    scan_text = self._extract_text_from_image(qp_region)
                    actual = self._get_qp_from_text(scan_text)
                    self.assertEqual(actual, _expected)

                except Exception as e:
                    self.fail(f'{impath}: {e}')

    def _extract_text_from_image(self, image):
        """
            via capy-drop-parser
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, qp_image = cv2.threshold(gray, 65, 255, cv2.THRESH_BINARY_INV)

        return pytesseract.image_to_string(
            qp_image,
            config="-l eng --oem 1 --psm 7 -c tessedit_char_whitelist=,0123456789",
        )

    def _get_qp_from_text(self, text):
        """
            via capy-drop-parser
        """
        qp = 0
        power = 1
        # re matches left to right so reverse the list
        # to process lower orders of magnitude first.
        for match in re.findall("[0-9]+", text)[::-1]:
            qp += int(match) * power
            power *= 1000

        return qp

    def test_guess_pageinfo_000(self):
        images_dir = get_images_absdir('000')
        expected = {
            '000.png': (1, 1, 0),
            '001.png': (1, 1, 0),
            '002.png': (1, 1, 3),
            '003.png': (1, 1, 3),
            '004.png': (1, 2, 4),
            '005.png': (2, 2, 4),
            '006.png': (1, 2, 6),
            '007.png': (2, 2, 6),
        }
        self._test_guess_pageinfo(images_dir, expected)

    def test_detect_qp_region_000(self):
        images_dir = get_images_absdir('000')
        expected = {
            '000.png': 746196407,
            '001.png': 746235637,
            '002.png': 3893778,
            '003.png': 17913226,
            '004.png': 17997826,
            '005.png': 17997826,
            '006.png': 563947174,
            '007.png': 563947174,
        }
        self._test_detect_qp_region(images_dir, expected)

    def test_guess_pageinfo_001(self):
        """
            いわゆる「イシュタル弓問題」のテスト。
            背景領域のオブジェクトをスクロールバーと誤認する問題
        """
        images_dir = get_images_absdir('001')
        expected = {
            '000.png': (1, 1, 3),
            '001.png': (1, 2, 4),
            '002.png': (2, 2, 4),
            '003.png': (1, 1, 3),
            '004.png': (1, 2, 4),
            '005.png': (2, 2, 4),
        }
        self._test_guess_pageinfo(images_dir, expected)

    def test_detect_qp_region_001(self):
        images_dir = get_images_absdir('001')
        expected = {
            '000.png': 477523200,
            '001.png': 75069014,
            '002.png': 75069014,
            '003.png': 75077524,
            '004.png': 75094324,
            '005.png': 75094324,
        }
        self._test_detect_qp_region(images_dir, expected)

    def test_guess_pageinfo_002(self):
        images_dir = get_images_absdir('002')
        expected = {
            '000.png': (2, 2, 5),
        }
        self._test_guess_pageinfo(images_dir, expected)

    def test_detect_qp_region_002(self):
        images_dir = get_images_absdir('002')
        expected = {
            '000.png': 981488422,
        }
        self._test_detect_qp_region(images_dir, expected)

    def test_guess_pageinfo_003(self):
        """
            2ページ目なのに3ページ目と判定される不具合を修正。
        """
        images_dir = get_images_absdir('003')
        expected = {
            '000.png': (2, 2, 6),
            '001.png': (1, 3, 7),
            '002.png': (2, 3, 7),
            '003.png': (3, 3, 7),
        }
        self._test_guess_pageinfo(images_dir, expected)

    def test_detect_qp_region_003(self):
        images_dir = get_images_absdir('003')
        expected = {
            '000.png': 999999999,
            '001.png': 229004798,
            '002.png': 229004798,
            '003.png': 229004798,
        }
        self._test_detect_qp_region(images_dir, expected)

    def test_guess_pageinfo_004(self):
        """
            スクロールバーの誤検出により認識エラーになる件について、
            スクロール可能領域を検出できない場合はスクロールバー
            なしと判定するようにした。
            https://github.com/max747/fgojunks/issues/1
        """
        images_dir = get_images_absdir('004')
        expected = {
            '000.png': (1, 1, 0),
        }
        self._test_guess_pageinfo(images_dir, expected)

    def test_detect_qp_region_004(self):
        images_dir = get_images_absdir('004')
        expected = {
            '000.png': 887919828,
        }
        self._test_detect_qp_region(images_dir, expected)

    def test_guess_pageinfo_005(self):
        """
            png だと正常に通るが jpg だと NG なケースについて、
            パラメータを修正して対応した。
            https://github.com/max747/fgojunks/issues/2
        """
        images_dir = get_images_absdir('005')
        expected = {
            '000.png': (2, 2, 4),
            '000.jpg': (2, 2, 4),
            '001.png': (2, 2, 4),
            '001.jpg': (2, 2, 4),
            '002.jpg': (1, 2, 4),
            '003.png': (1, 2, 4),
            '003.jpg': (1, 2, 4),
            '004.png': (1, 2, 4),
            '004.jpg': (1, 2, 4),
        }
        self._test_guess_pageinfo(images_dir, expected)

    def test_detect_qp_region_005(self):
        images_dir = get_images_absdir('005')
        expected = {
            '000.png': 333299361,
            '000.jpg': 333299361,
            '001.png': 764623092,
            '001.jpg': 764623092,
            '002.jpg': 813492669,
            '003.png': 331312461,
            '003.jpg': 331312461,
            '004.png': 338050061,
            '004.jpg': 338050061,
        }
        self._test_detect_qp_region(images_dir, expected)

    def test_guess_pageinfo_006(self):
        """
            閾値の設定が 26 以下ではスクロール可能領域の
            下端にヒゲが出てしまい矩形幅が広がってしまう jpg 画像。
        """
        images_dir = get_images_absdir('006')
        expected = {
            '000.jpg': (1, 2, 4),
        }
        self._test_guess_pageinfo(images_dir, expected)

    def test_detect_qp_region_006(self):
        images_dir = get_images_absdir('006')
        expected = {
            '000.jpg': 74780614,
        }
        self._test_detect_qp_region(images_dir, expected)

    def test_guess_pageinfo_007(self):
        """
            jpg 画像でイシュタル弓問題を含むスクロールバー誤検出が発生するケース

            000 イシュタル弓問題
                スクロールバー判定の閾値を 60 -> 61 に上げると解決する。
            001 スクロール可能領域をスクロールバーと誤検出
                スクロールバー判定の閾値を 64 以上に上げると解決する。
        """
        images_dir = get_images_absdir('007')
        expected = {
            '000.jpg': (1, 1, 3),
            '001.jpg': (1, 2, 5),
        }
        self._test_guess_pageinfo(images_dir, expected)

    def test_detect_qp_region_007(self):
        images_dir = get_images_absdir('007')
        expected = {
            '000.jpg': 328845347,
            '001.jpg': 172504749,
        }
        self._test_detect_qp_region(images_dir, expected)

    def test_guess_pageinfo_008(self):
        """
            jpg 画像でスクロール可能領域の検出がされにくいケース

            000 スクロール可能領域判定の閾値を 21 まで下げると解決する。
            001 スクロール可能領域判定の閾値を 18 まで下げると解決する。
        """
        images_dir = get_images_absdir('008')
        expected = {
            '000.jpg': (2, 2, 5),
            '001.jpg': (2, 2, 5),
        }
        self._test_guess_pageinfo(images_dir, expected)

    def test_detect_qp_region_008(self):
        images_dir = get_images_absdir('008')
        expected = {
            '000.jpg': 350300753,
            '001.jpg': 115078733,
        }
        self._test_detect_qp_region(images_dir, expected)

    def test_guess_pageinfo_009(self):
        """
            NA 版のスクリーンショットでエラーが出るケース page

            000 誤差の許容範囲を広げることで解決。
        """
        images_dir = get_images_absdir('009')
        pageinfo_expected = {
            '000.png': (1, 1, 3),
            '001.jpg': (2, 2, 4),
            '002.jpg': (1, 1, 0),
        }
        self._test_guess_pageinfo(images_dir, pageinfo_expected)

    def test_detect_qp_region_009(self):
        """
            NA 版のスクリーンショットでエラーが出るケース QP

            001 左下のボタンのせいでQP領域をうまく拾えない。解決不能。
                None が返されることを確認。
        """
        images_dir = get_images_absdir('009')
        qp_expected = {
            '000.png': 357256131,
            '001.jpg': None,
            '002.jpg': 243903289,
        }
        self._test_detect_qp_region(images_dir, qp_expected)

    def test_guess_pageinfo_010(self):
        """
            8行ドロップのスクリーンショット page
        """
        images_dir = get_images_absdir('010')
        pageinfo_expected = {
            '000.jpg': (1, 3, 7),
            '001.jpg': (2, 3, 7),
            '002.jpg': (3, 3, 7),
            '003.png': (1, 3, 8),
            '004.png': (2, 3, 8),
            '005.png': (3, 3, 8),
        }
        self._test_guess_pageinfo(images_dir, pageinfo_expected)

    def test_detect_qp_region_010(self):
        """
            8行ドロップのスクリーンショット QP
        """
        images_dir = get_images_absdir('010')
        qp_expected = {
            '000.jpg': 377361463,
            '001.jpg': 377361463,
            '002.jpg': 377361463,
            '003.png': 17643053,
            '004.png': 17643053,
            '005.png': 17643053,
        }
        self._test_detect_qp_region(images_dir, qp_expected)

    def test_guess_pageinfo_011(self):
        """
            タップ軌跡の影響で矩形がゆがむケース page
        """
        images_dir = get_images_absdir('011')
        pageinfo_expected = {
            '000.png': (1, 1, 3),
        }
        self._test_guess_pageinfo(images_dir, pageinfo_expected)

    def test_detect_qp_region_011(self):
        """
            タップ軌跡の影響で矩形がゆがむケース QP
        """
        images_dir = get_images_absdir('011')
        qp_expected = {
            '000.png': 288609041,
        }
        self._test_detect_qp_region(images_dir, qp_expected)

    def test_guess_pageinfo_012(self):
        """
            9行以上ドロップ page
        """
        images_dir = get_images_absdir('012')
        pageinfo_expected = {
            '62/000.jpg': (1, 3, 9),
            '62/001.jpg': (2, 3, 9),
            '62/002.jpg': (3, 3, 9),
            '64/000.jpg': (1, 4, 10),
            '64/001.jpg': (2, 4, 10),
            '64/002.jpg': (3, 4, 10),
            '64/003.jpg': (4, 4, 10),
            '72/000.jpg': (1, 4, 11),
            '72/001.jpg': (2, 4, 11),
            '72/002.jpg': (3, 4, 11),
            '72/003.jpg': (4, 4, 11),
            '82/000.jpg': (1, 4, 12),
            '82/001.jpg': (2, 4, 12),
            '82/002.jpg': (3, 4, 12),
            '82/003.jpg': (4, 4, 12),
            '90/000.jpg': (1, 5, 13),
            '90/001.jpg': (2, 5, 13),
            '90/002.jpg': (3, 5, 13),
            '90/003.jpg': (4, 5, 13),
            '90/004.jpg': (5, 5, 13),
            '93/000.jpg': (1, 5, 14),
            '93/001.jpg': (2, 5, 14),
            '93/002.jpg': (3, 5, 14),
            '93/003.jpg': (4, 5, 14),
            '93/004.jpg': (5, 5, 14),
            '98/000.jpg': (1, 5, 15),
            '98/001.jpg': (2, 5, 15),
            '98/002.jpg': (3, 5, 15),
            '98/003.jpg': (4, 5, 15),
            '98/004.jpg': (5, 5, 15),
            '100/000.jpg': (1, 5, 15),
            '100/001.jpg': (2, 5, 15),
            '100/002.jpg': (3, 5, 15),
            '100/003.jpg': (4, 5, 15),
            '100/004.jpg': (5, 5, 15),
        }
        self._test_guess_pageinfo(images_dir, pageinfo_expected)

    def test_detect_qp_region_012(self):
        """
            9行以上ドロップ qp
        """
        images_dir = get_images_absdir('012')
        pageinfo_expected = {
            '62/000.jpg': 65638587,
            '62/001.jpg': 65638587,
            '62/002.jpg': 65638587,
            '64/000.jpg': 65736387,
            '64/001.jpg': 65736387,
            '64/002.jpg': 65736387,
            '64/003.jpg': 65736387,
            '72/000.jpg': 65743787,
            '72/001.jpg': 65743787,
            '72/002.jpg': 65743787,
            '72/003.jpg': 65743787,
            '82/000.jpg': 71925747,
            '82/001.jpg': 71925747,
            '82/002.jpg': 71925747,
            '82/003.jpg': 71925747,
            '90/000.jpg': 68906947,
            '90/001.jpg': 68906947,
            '90/002.jpg': 68906947,
            '90/003.jpg': 68906947,
            '90/004.jpg': 68906947,
            '93/000.jpg': 90349741,
            '93/001.jpg': 90349741,
            '93/002.jpg': 90349741,
            '93/003.jpg': 90349741,
            '93/004.jpg': 90349741,
            '98/000.jpg': 94449841,
            '98/001.jpg': 94449841,
            '98/002.jpg': 94449841,
            '98/003.jpg': 94449841,
            '98/004.jpg': 94449841,
            # 100/*.jpg は解像度が低いため枠を検出できない
        }
        self._test_detect_qp_region(images_dir, pageinfo_expected)

    def test_guess_pageinfo_013(self):
        """
            村正のマントでスクロール可能領域が縦に広がるケース page
        """
        images_dir = get_images_absdir('013')
        pageinfo_expected = {
            '000.jpg': (1, 1, 3),
            '001.jpg': (1, 1, 3),
        }
        self._test_guess_pageinfo(images_dir, pageinfo_expected)

    def test_detect_qp_region_013(self):
        """
            村正のマントでスクロール可能領域が縦に広がるケース QP
        """
        images_dir = get_images_absdir('013')
        qp_expected = {
            '000.jpg': 993204351,
            '001.jpg': 993247551,
        }
        self._test_detect_qp_region(images_dir, qp_expected)

    def test_guess_pageinfo_014(self):
        """
            背景の木をスクロールバーと判定してしまうケース page
        """
        images_dir = get_images_absdir('014')
        pageinfo_expected = {
            '000.jpg': (1, 1, 0),
            '001.jpg': (1, 1, 0),
        }
        self._test_guess_pageinfo(images_dir, pageinfo_expected)

    def test_detect_qp_region_014(self):
        """
            背景の木をスクロールバーと判定してしまうケース QP
        """
        images_dir = get_images_absdir('014')
        qp_expected = {
            '000.jpg': 404036268,
            '001.jpg': 403885798,
        }
        self._test_detect_qp_region(images_dir, qp_expected)

    def test_guess_pageinfo_015(self):
        """
            左右に黒余白がある (new iPad mini) page
        """
        images_dir = get_images_absdir('015')
        pageinfo_expected = {
            '000.jpg': (1, 1, 3),
        }
        self._test_guess_pageinfo(images_dir, pageinfo_expected)

    def test_detect_qp_region_015(self):
        """
            左右に黒余白がある (new iPad mini) QP
        """
        images_dir = get_images_absdir('015')
        qp_expected = {
            '000.jpg': 1106700890,
        }
        self._test_detect_qp_region(images_dir, qp_expected)

    def test_guess_pageinfo_016(self):
        """
            QP領域の検出でフチを矩形として検出してしまう page
        """
        images_dir = get_images_absdir('016')
        pageinfo_expected = {
            '000.jpg': (1, 2, 4),
        }
        self._test_guess_pageinfo(images_dir, pageinfo_expected)

    def test_detect_qp_region_016(self):
        """
            QP領域の検出でフチを矩形として検出してしまう qp
        """
        images_dir = get_images_absdir('016')
        qp_expected = {
            '000.jpg': 1869090954,
        }
        self._test_detect_qp_region(images_dir, qp_expected)

    def test_guess_pageinfo_017(self):
        """
            threshold を超えて line: 4 と判定されてしまう page
        """
        images_dir = get_images_absdir('017')
        pageinfo_expected = {
            '000.jpg': (1, 1, 3),
            '001.jpg': (1, 1, 3),
        }
        self._test_guess_pageinfo(images_dir, pageinfo_expected)

    def test_detect_qp_region_017(self):
        """
            threshold を超えて line: 4 と判定されてしまう qp
        """
        images_dir = get_images_absdir('017')
        qp_expected = {
            '000.jpg': 1981323944,
            '001.jpg': 2000000000,
        }
        self._test_detect_qp_region(images_dir, qp_expected)

    def test_guess_pageinfo_018(self):
        """
            wide screen の場合にスクロールバーが検出されない page
        """
        images_dir = get_images_absdir('018')
        pageinfo_expected = {
            '000.png': (1, 2, 4),
            '001.png': (2, 2, 4),
        }
        self._test_guess_pageinfo(images_dir, pageinfo_expected)

    def test_detect_qp_region_018(self):
        """
            wide screen の場合にスクロールバーが検出されない qp
        """
        images_dir = get_images_absdir('018')
        qp_expected = {
            '000.png': 1156319000,
            '001.png': 1156319000,
        }
        self._test_detect_qp_region(images_dir, qp_expected)

    def test_guess_pageinfo_019(self):
        """
            千年城背景の影響でスクロールバーがジャギーになり頂点数が許容値を超える page
        """
        images_dir = get_images_absdir('019')
        pageinfo_expected = {
            '000.jpg': (2, 2, 6),
        }
        self._test_guess_pageinfo(images_dir, pageinfo_expected)

    def test_detect_qp_region_019(self):
        """
            千年城背景の影響でスクロールバーがジャギーになり頂点数が許容値を超える qp
        """
        images_dir = get_images_absdir('019')
        qp_expected = {
            '000.jpg': 24755544,
        }
        self._test_detect_qp_region(images_dir, qp_expected)

    def test_guess_pageinfo_020(self):
        """
            右端が黒背景の場合にスクロールバーが検出されない page
        """
        images_dir = get_images_absdir('020')
        pageinfo_expected = {
            '000.png': (1, 2, 4),
            '001.png': (2, 2, 4),
            '002.png': (1, 2, 4),
            '003.png': (2, 2, 4),
        }
        self._test_guess_pageinfo(images_dir, pageinfo_expected)

    def test_detect_qp_region_020(self):
        """
            右端が黒背景の場合にスクロールバーが検出されない qp
        """
        images_dir = get_images_absdir('020')
        qp_expected = {
            '000.png': 1970152495,
            '001.png': 1970152495,
            '002.png': 1970155395,
            '003.png': 1970155395,
        }
        self._test_detect_qp_region(images_dir, qp_expected)
