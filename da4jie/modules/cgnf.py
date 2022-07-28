from cProfile import label
from itertools import combinations

import torch
import torch.nn as nn

from .module_utils import *
from .crf import CRF

class CGNF(nn.Module):
    def __init__(self, label_vocab, bioes=False, max_len=512):
        super().__init__()

        self.label_vocab = label_vocab
        #QUES: why + 2 ?  #ANSW: S and E sequence tag (special toks)
        # self.label_size = len(label_vocab) + 2
        self.label_size = len(label_vocab)
        self.bioes = bioes

        # self.start = self.label_size - 2
        # self.end = self.label_size - 1
        transition = torch.randn(self.label_size, self.label_size)
        self.transition = nn.Parameter(transition)
        # self.max_len = max_len
        self.initialize()

    def initialize(self):
        # -100 ~ ignore by CrossEntLoss
        # self.transition.data[:, self.end] = -100.0
        # self.transition.data[self.start, :] = -100.0

        # for label, label_idx in self.label_vocab.items():
        #     if label.startswith('I-') or label.startswith('E-'):
        #         # Inside to End
        #         self.transition.data[label_idx, self.start] = -100.0
        #     if label.startswith('B-') or label.startswith('I-'):
        #         # Begin to Inside
        #         self.transition.data[self.end, label_idx] = -100.0

        # Iter twice to define constraint between consecutive labels
        for label_from, label_from_idx in self.label_vocab.items():
            if label_from == 'O':
                label_from_prefix, label_from_type = 'O', 'O'
            else:
                label_from_prefix, label_from_type = label_from.split('-', 1)
        
            for label_to, label_to_idx in self.label_vocab.items():
                if label_to == 'O':
                    label_to_prefix, label_to_type = 'O', 'O'
                else:
                    label_to_prefix, label_to_type = label_to.split('-', 1)

                # define contraints
                if self.bioes:
                    is_allowed = any(
                        [
                            label_from_prefix in ['O', 'E', 'S']
                            and label_to_prefix in ['O', 'B', 'S'],

                            label_from_prefix in ['B', 'I']
                            and label_to_prefix in ['I', 'E']
                            and label_from_type == label_to_type
                        ]
                    )
                else:
                    is_allowed = any(
                        # from > to: _-_ > B-_ ; _-_ > 0 ;  B-x > I-x ; I-x > I-x
                        [
                            label_to_prefix in ['B', 'O'],

                            label_from_prefix in ['B', 'I']
                            and label_to_prefix == 'I'
                            and label_from_type == label_to_type
                        ]
                    )
                if not is_allowed:
                    # from > to : B-x > I-y ; O > I-_ ; I-x > I-y
                    self.transition.data[label_to_idx, label_from_idx] = -100.0

    # def pad_logits(self, logits):
    #     #! remove
    #     """Pad the linear layer output with <SOS> and <EOS> scores.
    #     :param logits: Linear layer output (no non-linear function).
    #     """
    #     batch_size, seq_len, _ = logits.size()
    #     pads = logits.new_full((batch_size, seq_len, 2), -100.0,
    #                            requires_grad=False)
    #     #QUES: why SOS and EOS together at the end ? #ANSW: of labels seq, not tok seq, whose pos do not matter 
    #     logits = torch.cat([logits, pads], dim=2)
    #     return logits

    def _calc_binary_score_full(self,labels, len, A):
        #! Iter all (yi,yj) , sum Aij x Uij
        batch_size, seq_len = labels.size()

        pos_pairs = combinations(range(seq_len), 2)
        label_pairs = [combinations(label, 2) for label in labels]

        # # A tensor of size batch_size * (seq_len + 2)
        # #! new_empty/full preserve device, dtype
        # labels_ext = labels.new_empty((batch_size, seq_len + 2)) 
        # labels_ext[:, 0] = self.start  # Add SOS at start
        # labels_ext[:, 1:-1] = labels
        # mask = sequence_mask(lens + 1, max_len=(seq_len + 2)).long()
        # pad_stop = labels.new_full((1,), self.end, requires_grad=False)
        # pad_stop = pad_stop.unsqueeze(-1).expand(batch_size, seq_len + 2)
        # #QUES: same as labels.new_full((batch_size, seq_len + 2), self.end, requires_grad=False)
        # labels_ext = (1 - mask) * pad_stop + mask * labels_ext  # Add EOS to the end according to each seq len
        # labels = labels_ext  # bsz * seq_len+2



        trn = self.transition
        trn_exp = trn.unsqueeze(0).expand(batch_size, self.label_size, self.label_size)
        lbl_r = labels[:, :]
        lbl_rexp = lbl_r.unsqueeze(-1).expand(*lbl_r.size(), self.label_size)  # bsz * seq_len+1 * lbl_sz
        # score of jumping to a tag
        # out[i][j][k] = input[i] [index[i][j][k]] [k]  # if dim == 1
        trn_row = torch.gather(trn_exp, 1, lbl_rexp)  # bsz * seq_len+1 * lbl_sz

        lbl_lexp = labels[:, :-1].unsqueeze(-1)  # bsz * seq_len+1 * 1
        # out[i][j][k] = input[i] [j] [index[i][j][k]]  # if dim == 2
        trn_scr = torch.gather(trn_row, 2, lbl_lexp)  # bsz * seq_len+1 * 1
        trn_scr = trn_scr.squeeze(-1)

        mask = sequence_mask(lens + 1, max_len=trn_scr.size(-1)).float()  # bsz * seq_len+1
        trn_scr = trn_scr * mask
        score = trn_scr
        return score

    def _calc_binary_score_pseudo(self, labels, len, A):
        #! Iter all ( y_i,y_n(i) ) , sum Ai,n(i) x Ui,n(i)

    def calc_binary_score(self, labels, lens, A):
        if self.mode == 'pseudo':
            return self._calc_binary_score_pseudo(labels, len, A)
        
        elif self.mode == 'full':
            return self._calc_binary_score_full(labels, len, A)

    def calc_unary_score(self, logits, labels, lens):
        # ∑ ff(x_i)
        # print(lens)
        # labels_exp = labels.unsqueeze(-1)
        labels_exp = torch.clone(labels).unsqueeze(-1)
        # Convert -100 to 0 ( mask will handle the ignore instead of -100)
        labels_exp[labels_exp<0] = 0
        scores = torch.gather(logits, 2, labels_exp).squeeze(-1)
        # print(scores.size())
        mask = sequence_mask(lens.int(), max_len=scores.size(-1))
        # print(mask.size())
        mask = mask.float()
        scores = scores * mask
        return scores

    def calc_gold_score(self, logits, labels, lens, A):
        # s(X,z)
        unary_score = self.calc_unary_score(logits, labels, lens).sum(
            1).squeeze(-1)
        binary_score = self.calc_binary_score(labels, lens, A).sum(1).squeeze(-1)
        return unary_score + binary_score

    def calc_norm_score(self, logits, lens):
        # log ∑_i e^s(X,z_i)
        batch_size, _, _ = logits.size()
        alpha = logits.new_full((batch_size, self.label_size), -100.0)
        alpha[:, self.start] = 0
        lens_ = lens.clone()

        logits_t = logits.transpose(1, 0)
        for logit in logits_t:
            logit_exp = logit.unsqueeze(-1).expand(batch_size,
                                                   self.label_size,
                                                   self.label_size)
            alpha_exp = alpha.unsqueeze(1).expand(batch_size,
                                                  self.label_size,
                                                  self.label_size)
            trans_exp = self.transition.unsqueeze(0).expand_as(alpha_exp)
            mat = logit_exp + alpha_exp + trans_exp
            alpha_nxt = log_sum_exp(mat, 2).squeeze(-1)

            mask = (lens_ > 0).float().unsqueeze(-1).expand_as(alpha)
            alpha = mask * alpha_nxt + (1 - mask) * alpha
            lens_ = lens_ - 1

        alpha = alpha + self.transition[self.end].unsqueeze(0).expand_as(alpha)
        norm = log_sum_exp(alpha, 1).squeeze(-1)

        return norm

    def loglik(self, logits, labels, lens, A):
        # log P(z|X) = s(X,z) - log ∑_i e^s(X,z_i)
        norm_score = self.calc_norm_score(logits, lens)
        gold_score = self.calc_gold_score(logits, labels, lens, A)
        return gold_score - norm_score

    def viterbi_decode(self, logits, lens):
        """
            https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Sequence-Labeling#viterbi-decoding
            logits: [batch_size, seq_len, n_labels] FloatTensor
            lens: [batch_size] LongTensor
        """
        batch_size, _, n_labels = logits.size()
        vit = logits.new_full((batch_size, self.label_size), -100.0) 
        vit[:, self.start] = 0 # all -100 but first row SOS = 0
        c_lens = lens.clone()

        logits_t = logits.transpose(1, 0)
        pointers = []  # backpointer
        for logit in logits_t:
            vit_exp = vit.unsqueeze(1).expand(batch_size, n_labels, n_labels)  # curr scores
            trn_exp = self.transition.unsqueeze(0).expand_as(vit_exp)  # cumu scores
            vit_trn_sum = vit_exp + trn_exp 
            vt_max, vt_argmax = vit_trn_sum.max(2)  # get prev tag that max at each curr tag (bsz * prev * curr)

            vt_max = vt_max.squeeze(-1)
            vit_nxt = vt_max + logit
            pointers.append(vt_argmax.squeeze(-1).unsqueeze(0))

            # stop update for seq that reach EOS
            mask = (c_lens > 0).float().unsqueeze(-1).expand_as(vit_nxt) 
            vit = mask * vit_nxt + (1 - mask) * vit

            # last token EOS 
            mask = (c_lens == 1).float().unsqueeze(-1).expand_as(vit_nxt)
            vit += mask * self.transition[self.end].unsqueeze(
                0).expand_as(vit_nxt)

            c_lens = c_lens - 1

        pointers = torch.cat(pointers)
        scores, idx = vit.max(1)
        paths = [idx.unsqueeze(1)]
        for argmax in reversed(pointers):
            idx_exp = idx.unsqueeze(-1)
            idx = torch.gather(argmax, 1, idx_exp)
            idx = idx.squeeze(-1)

            paths.insert(0, idx.unsqueeze(1))

        paths = torch.cat(paths[1:], 1)
        scores = scores.squeeze(-1)

        return scores, paths

    def calc_conf_score_(self, logits, labels):
        batch_size, _, _ = logits.size()

        logits_t = logits.transpose(1, 0)
        scores = [[] for _ in range(batch_size)]
        pre_labels = [self.start] * batch_size
        for i, logit in enumerate(logits_t):
            logit_exp = logit.unsqueeze(-1).expand(batch_size,
                                                   self.label_size,
                                                   self.label_size)
            trans_exp = self.transition.unsqueeze(0).expand(batch_size,
                                                            self.label_size,
                                                            self.label_size)
            score = logit_exp + trans_exp
            score = score.view(-1, self.label_size * self.label_size) \
                .softmax(1)
            for j in range(batch_size):
                cur_label = labels[j][i]
                cur_score = score[j][cur_label * self.label_size + pre_labels[j]]
                scores[j].append(cur_score)
                pre_labels[j] = cur_label
        return scores