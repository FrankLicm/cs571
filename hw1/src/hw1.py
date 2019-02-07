# ========================================================================
# Copyright 2019 ELIT
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ========================================================================
import csv
import os
import glob
import re
from typing import List, Any
from elit.component import Component

__author__ = "Gary Lai, Jinho D. Choi"


class HashtagSegmenter(Component):

    def __init__(self, resource_dir: str):
        """
        :param resource_dir: a path to the directory where resource files are located.
        """
        # initialize the n-grams
        ngram_filenames = glob.glob(os.path.join(resource_dir, '[1-2]gram.txt'))
        two_char_word_file = glob.glob(os.path.join(resource_dir, 'twocharword.txt'))
        self.two_char_words = []
        for filename in two_char_word_file:
            with open(filename) as st:
                for line in st:
                    self.two_char_words.append(line.strip("\n"))
        print(self.two_char_words)
        self.parsedDict = {}
        # TODO: initialize resources
        self.n_grams = [{}, {}, {}, {}, {}, {}]
        self.n_grams_total_appearance_count = [0, 0, 0, 0, 0, 0]
        self.n_grams_total_word_count = [0, 0, 0, 0, 0, 0]
        self.n_grams_minimum_appearance_count = [0, 0, 0, 0, 0, 0]
        assert (len(ngram_filenames) > 0, "there is no grams files, please add them to res directory")
        for filename in ngram_filenames:
            print(filename)
            gram_n = int(filename.split("/")[-1].split("gram")[0])
            print(filename.split("/")[2].split("gram")[0])
            n_gram_dict = {}
            total_count = 0
            total_word_count = 0
            minimum = float("inf")
            with open(filename) as st:
                for line in st:
                    gram_type = line.split("\t")
                    n_gram_dict[gram_type[1].strip("\n")] = int(gram_type[0])
                    total_count += int(gram_type[0])
                    total_word_count += 1
                    if int(gram_type[0]) < minimum:
                        minimum = int(gram_type[0])
            self.n_grams[gram_n-1] = n_gram_dict
            self.n_grams_total_appearance_count[gram_n-1] = total_count
            self.n_grams_total_word_count[gram_n-1] = total_word_count
            self.n_grams_minimum_appearance_count[gram_n-1] = minimum

        # convert bi-gram dict to dict whose key is the first token and value is minimum appearance count among that token)
        self.bi_gram_minimum_appearance_dict = {}
        for gram in self.n_grams[1]:
            words = gram.split(" ")
            if words[0] in self.bi_gram_minimum_appearance_dict:
                if self.n_grams[1][gram] < self.bi_gram_minimum_appearance_dict[words[0]]:
                    self.bi_gram_minimum_appearance_dict[words[0]] = self.n_grams[1][gram]
            else:
                self.bi_gram_minimum_appearance_dict[words[0]] = self.n_grams[1][gram]

        self.bi_gram_token_count = {}
        for gram in self.n_grams[1]:
            words = gram.split(" ")
            if words[0] in self.bi_gram_token_count:
                    self.bi_gram_token_count[words[0]] += 1
            else:
                self.bi_gram_token_count[words[0]] = 1

    def decode(self, hashtag: str, **kwargs) -> List[str]:
        """
        :param hashtag: the input hashtag starting with `#` (e.g., '#helloworld').
        :param kwargs:
        :return: the list of tokens segmented from the hashtag (e.g., ['hello', 'world']).
        """
        # TODO: update the following code.
        text = hashtag[1:]
        results = []
        # remember the case
        upper_index = []
        for i, character in enumerate(text):
            if character.isupper():
                upper_index.append(i)

        # deal with upper case and number
        if (len(upper_index) >= 2 and self.check_continues(upper_index)) or (len(upper_index) == 1 and upper_index[0] != 0):
            final_results = []
            try:
                results = self.segment_based_on_cap_and_number(text, upper_index)
                for result in results:
                    if result.isdigit():
                        final_results.append(result)
                    else:
                        final_results = final_results + self.deal_with_text(result)
                return self.recover_the_case(final_results, upper_index)
            except:
                pass

        # deal with long hash tag
        if len(text) > 22:
            text = text.lower()
            results = self.deal_with_long_hashtag(text,0)
            return self.recover_the_case(results, upper_index)

        # deal with "_" symbol
        if "_" in text:
            final_results = []
            texts = text.split("_")
            for text in texts:
                results.append(text)
                results.append("_")
            for result in results:
                if len(result) <= 1:
                    final_results.append(result)
                else:
                    final_results = final_results + self.deal_with_text(result)
            return self.recover_the_case(final_results[:-1], upper_index)

        # deal with number
        if re.search(r"\d+", text):
            number_list = []
            non_number_list = []
            final_results = []
            for char in text:
                if char == 'O' or char.isdigit():
                    if len(non_number_list) > 0:
                        results.append("".join(non_number_list))
                        non_number_list.clear()
                    number_list.append(char)
                else:
                    if len(number_list) > 0:
                        results.append("".join(number_list))
                        number_list.clear()
                    non_number_list.append(char)
            try:
                results.append(text.split(results[-1])[-1])
                for result in results:
                    if re.search(r"\d+", result):
                        final_results.append(result)
                    else:
                        final_results = final_results + self.deal_with_text(result)
                real_results = []
                less_than_two_list = []
                for result in final_results:
                    if len(final_results) > 1:
                        if len(result) <= 2 and result not in ["i", "a"] and result not in self.two_char_words:
                            less_than_two_list.append(result)
                        else:
                            if len(less_than_two_list) > 0:
                                real_results.append("".join(less_than_two_list))
                                less_than_two_list = []
                            real_results.append(result)
                if len(less_than_two_list) > 0:
                    real_results.append("".join(less_than_two_list))
                if real_results:
                    final_results = real_results
                return self.recover_the_case(final_results, upper_index)
            except:
                pass

        # not all of above
        results = self.deal_with_text(text)
        # recover the case
        return self.recover_the_case(results, upper_index)

    def deal_with_long_hashtag(self, text, i):
        if len(text) == 0:
            return []
        index = len(text) - 1
        segmented = []
        while len(text) - 1 >= 0:
            seperated = text[:index + 1]
            if seperated in ["a", "i"] or seperated in self.two_char_words or (len(seperated)>2 and seperated in self.n_grams[0]):
                segmented.append(seperated)
                if len(text[index + 1:]) < 23:
                    rem = self.deal_with_text(text[index + 1:])
                else:
                    rem = self.deal_with_long_hashtag(text[index + 1:], i+1)
                if len(rem) > 0:
                    segmented = segmented + rem
                return segmented
            index = index - 1
        segmented.append(text[0])
        rem = text(text[1:])
        tokens = segmented + rem
        return tokens

    def recover_the_case(self, results: list, upper_index: list):
        # recover the case
        start = 0
        last_end = -1
        for i, result in enumerate(results):
            result_list = list(result)
            word_len = len(result)
            end = start + word_len - 1
            for index in upper_index:
                if index > end:
                    break
                if index <= last_end:
                    continue
                result_list[index - start] = result[index - start].upper()
            start = start + len(result)
            last_end = end
            results[i] = "".join(result_list)
        return results

    def deal_with_text(self, text):
        # normalize to lower case
        text = text.lower()

        # calculate the result based on 1-2gram
        segmentation_list = self.generate_possible_segmentation(text)
        if text not in self.n_grams[0]:
            results = self.get_results(segmentation_list)

            if len(results) == 0:
                if re.search(r"\d+", text):
                    number_list = []
                    non_number_list = []
                    for char in text:
                        if char.isdigit():
                            if len(non_number_list) > 0:
                                results.append("".join(non_number_list))
                                non_number_list.clear()
                            number_list.append(char)
                        else:
                            if len(number_list) > 0:
                                results.append("".join(number_list))
                                number_list.clear()
                            non_number_list.append(char)
                    try:
                        results.append(text.split(results[-1])[-1])
                    except:
                        self.deal_with_unknown(segmentation_list)
                else:
                    results = self.deal_with_unknown(segmentation_list)

            real_results = []
            less_than_two_list = []
            for result in results:
                if len(results) > 1:
                    if len(result) <= 2 and result not in ["i", "a"] and result not in self.two_char_words and result.isdigit() is False:
                        less_than_two_list.append(result)
                    else:
                        if len(less_than_two_list) > 0:
                            real_results.append("".join(less_than_two_list))
                            less_than_two_list = []
                        real_results.append(result)
            if len(less_than_two_list) > 0:
                real_results.append("".join(less_than_two_list))
            if real_results:
                results = real_results

            # filter one
            known_flag = 0
            less_than_two_flag = 0
            for result in results:
                if len(results) > 1:
                    if len(result) == 2 and result not in self.two_char_words and result.isdigit() is False:
                        less_than_two_flag += 1
                    elif len(result) > 4 and result in self.n_grams[0]:
                        known_flag += 1
            if known_flag == 0 and less_than_two_flag >= 1:
                return [text]

            # filter two
            known_flag = 0
            less_than_two_flag = 0
            for result in results:
                if len(results) > 1:
                    if len(result) <= 1 and result not in ["i", "a"]:
                        less_than_two_flag += 1
                    elif len(result) > 5 and result in self.n_grams[0]:
                        known_flag += 1
            if known_flag == 0 and less_than_two_flag >= 1:
                return [text]

        else:
            results = [text]
        return results

    def check_continues(self, index_list: list):
        for i, index in enumerate(index_list):
            if i ==0:
                continue
            else:
                if index-index_list[i-1] == 1:
                    return False
        return True

    def check_available(self, segmentation: list):
        flags = {}
        flags[True] = 0
        flags[False] = 0
        for seg in segmentation:
            if seg not in self.n_grams[0] or seg not in self.bi_gram_token_count:
                return False
        return True

    def deal_with_unknown(self, segmentation_list: list):
        maxlength = 0
        results = []
        max_known_number = 0
        for segmentation in segmentation_list:
            known_number = 0
            tmp_maxlength = 0
            for seg in segmentation:
                if seg in self.n_grams[0]:
                    known_number += 1
                    if len(seg) > maxlength:
                        tmp_maxlength = len(seg)
                if known_number >= max_known_number and tmp_maxlength >= maxlength:
                    max_known_number = known_number
                    maxlength = tmp_maxlength
                    results = segmentation
        return results

    def segment_based_on_cap_and_number(self, text: str, upper_index: list):
        results = []
        if upper_index[0] == 0:
            for i, index in enumerate(upper_index):
                if i == 0:
                    continue
                else:
                    results.append(text[upper_index[i - 1]:upper_index[i]])
        else:
            for i, index in enumerate(upper_index):
                if i == 0:
                    results.append(text[0:upper_index[i]])
                else:
                    results.append(text[upper_index[i - 1]:upper_index[i]])
        results.append(text[upper_index[-1]:len(text)])
        results = self.deal_with_number(results)
        return results

    def deal_with_number(self, results):
        new_results = []
        for result in results:
            if not result[-1].isdigit():
                new_results.append(result)
            else:
                digit_list = []
                i = len(result) - 1
                while i > 0:
                    if result[i].isdigit():
                        digit_list.append(result[i])
                    else:
                        break
                    i -= 1
                digit_list.reverse()
                digits = "".join(digit_list)
                new_results.append(result.split(digits)[0])
                new_results.append(digits)
        if new_results:
            return new_results
        return results

    def get_results(self, segmentation_list:list):
        max_prob = 0
        results = []

        for segmentation in segmentation_list:
            if self.check_available(segmentation):
                prob = self.compute_segmentation_probability_by_discount_smoothing(segmentation)
                if prob > max_prob:
                    results = segmentation
                    max_prob = prob

        return results

    def generate_possible_segmentation(self, text: str):
        segmentation_list = []
        head_tails = [(text[:i + 1], text[i + 1:]) for i in range(len(text))]
        for head, tail in head_tails:
            if len(tail) > 0:
                sub_segmentation_list = self.generate_possible_segmentation(tail)
                if len(sub_segmentation_list) > 0:
                    for sub_segmentations in sub_segmentation_list:
                        new_segmentation = [head]
                        for sub_segmentation in sub_segmentations:
                            new_segmentation += [sub_segmentation]
                        segmentation_list.append(new_segmentation)

            else:
                segmentation_list.append([head])
        return segmentation_list

    def compute_segmentation_probability_by_discount_smoothing(self, new_segmentation: list):
        # p(x0)p(x1|x0)p(x2|x1)p(x3|x2)...
        #  discount smoothing parameter a
        a = 0.7
        if len(new_segmentation) <= 0:
            return
        prob_result = 1
        # for a word, we need record its subtract prob(list) in n-gram
        two_gram_subtract_prob_list = {}

        one_gram_minimum_appearance_time = self.n_grams_minimum_appearance_count[0]
        for i,segmentation in enumerate(new_segmentation):
            if i == 0:
                if segmentation in self.n_grams[0]:
                    prob_result = prob_result * float(self.n_grams[0][segmentation])/self.n_grams_total_appearance_count[0]
                # Jinho's discount smoothing for uni-gram
                else:
                    prob = a * float(one_gram_minimum_appearance_time) / self.n_grams_total_appearance_count[0]
                    prob_result = prob_result * prob
                    one_gram_minimum_appearance_time = a * one_gram_minimum_appearance_time
                continue

            bisegmentation = new_segmentation[i-1]+" "+segmentation
            if bisegmentation in self.n_grams[1]:
                #print(bisegmentation)
                new_count = float(self.n_grams[1][bisegmentation])
                if new_segmentation[i-1] in two_gram_subtract_prob_list:
                    for prob in two_gram_subtract_prob_list[new_segmentation[i-1]]:
                        new_count = float(new_count) - float(prob)
                prob_result = 100*prob_result * new_count/(self.n_grams[0][new_segmentation[i-1]])

            # discount smoothing for bi-gram
            else:
                if new_segmentation[i-1] in self.bi_gram_minimum_appearance_dict:
                    new_count = a * self.bi_gram_minimum_appearance_dict[new_segmentation[i-1]]
                    prob = float(new_count)/(self.n_grams[0][new_segmentation[i-1]])
                    prob_result = prob_result * prob
                    #print("new_segmentation[i-1]:", new_segmentation[i-1])
                    #print("two_gram_subtract_prob_list:",two_gram_subtract_prob_list)
                    if new_segmentation[i-1] in two_gram_subtract_prob_list:
                        prob_list = two_gram_subtract_prob_list[new_segmentation[i-1]]
                        prob_list.append(prob)
                        two_gram_subtract_prob_list[new_segmentation[i-1]] = prob_list
                    else:
                        two_gram_subtract_prob_list[new_segmentation[i-1]] = [prob]
                else:
                    new_count = a * self.n_grams_minimum_appearance_count[1]
                    if new_segmentation[i-1] in self.n_grams[0]:
                        prob = float(new_count)/self.n_grams[0][new_segmentation[i-1]]
                        prob_result = prob_result * prob
                        if new_segmentation[i - 1] in two_gram_subtract_prob_list:
                            prob_list = two_gram_subtract_prob_list[new_segmentation[i - 1]]
                            prob_list.append(prob)
                            two_gram_subtract_prob_list[new_segmentation[i - 1]] = prob_list
                        else:
                            two_gram_subtract_prob_list[new_segmentation[i - 1]] = [prob]
                    else:
                        new_uni_count = a*one_gram_minimum_appearance_time
                        one_gram_minimum_appearance_time = a * one_gram_minimum_appearance_time
                        prob = float(new_count)/new_uni_count
                        prob_result = prob_result * prob
                        if new_segmentation[i - 1] in two_gram_subtract_prob_list:
                            prob_list = two_gram_subtract_prob_list[new_segmentation[i - 1]]
                            prob_list.append(prob)
                            two_gram_subtract_prob_list[new_segmentation[i - 1]] = prob_list
                        else:
                            two_gram_subtract_prob_list[new_segmentation[i - 1]] = [prob]
        return prob_result

    def compute_segmentation_probability_by_add_k_smoothing(self, new_segmentation: list):
        # p(x0)p(x1|x0)p(x2|x1)p(x3|x2)...
        prob_result = 1
        k = 23
        if len(new_segmentation) <= 0:
            return 0
        for i, segmentation in enumerate(new_segmentation):
            if i == 0:
                if segmentation in self.n_grams[0]:
                    prob_result = prob_result * float(self.n_grams[0][segmentation])/self.n_grams_total_appearance_count[0]
            pre_x = new_segmentation[i-1]
            cur_x = segmentation
            bisegmentation = pre_x + " " + cur_x
            if bisegmentation in self.n_grams[1]:
                new_count = float(self.n_grams[1][bisegmentation])
                prob_result = 10*prob_result * new_count / (self.n_grams[0][pre_x])
            else:
                prob_result = prob_result * k/(self.n_grams[0][pre_x] + k*self.bi_gram_token_count[pre_x])
        return prob_result

    def evaluate(self, data: Any, **kwargs):
        pass  # NO NEED TO UPDATE

    def load(self, model_path: str, **kwargs):
        pass  # NO NEED TO UPDATE

    def save(self, model_path: str, **kwargs):
        pass  # NO NEED TO UPDATE

    def train(self, trn_data, dev_data, *args, **kwargs):
        pass  # NO NEED TO UPDATE


if __name__ == '__main__':
    resource_dir = "../res"
    segmenter = HashtagSegmenter(resource_dir)
    total = correct = 0
    with open(os.path.join(resource_dir, 'hashtags.csv')) as fin:
        reader = csv.reader(fin)
        for row in reader:
            hashtag = row[0]
            gold = row[1]
            auto = ' '.join(segmenter.decode(hashtag))
            print('%s -> %s | %s' % (hashtag, auto, gold))
            if gold == auto:correct += 1
            total += 1
    print('%5.2f (%d/%d)' % (100.0*correct/total, correct, total))
