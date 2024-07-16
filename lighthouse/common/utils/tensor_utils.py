"""
Copyright $today.year LY Corporation

LY Corporation licenses this file to you under the Apache License,
version 2.0 (the "License"); you may not use this file except in compliance
with the License. You may obtain a copy of the License at:

  https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
License for the specific language governing permissions and limitations
under the License.

Moment-DETR (https://github.com/jayleicn/moment_detr)
Copyright (c) 2021 Jie Lei

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import numpy as np
import torch


def pad_sequences_1d(sequences, dtype=torch.long, device=torch.device("cpu"), fixed_length=None):
    """ Pad a single-nested list or a sequence of n-d array (torch.tensor or np.ndarray)
    into a (n+1)-d array, only allow the first dim has variable lengths.
    Args:
        sequences: list(n-d tensor or list)
        dtype: np.dtype or torch.dtype
        device:
        fixed_length: pad all seq in sequences to fixed length. All seq should have a length <= fixed_length.
            return will be of shape [len(sequences), fixed_length, ...]
    Returns:
        padded_seqs: ((n+1)-d tensor) padded with zeros
        mask: (2d tensor) of the same shape as the first two dims of padded_seqs,
              1 indicate valid, 0 otherwise
    Examples:
        >>> test_data_list = [[1,2,3], [1,2], [3,4,7,9]]
        >>> pad_sequences_1d(test_data_list, dtype=torch.long)
        >>> test_data_3d = [torch.randn(2,3,4), torch.randn(4,3,4), torch.randn(1,3,4)]
        >>> pad_sequences_1d(test_data_3d, dtype=torch.float)
        >>> test_data_list = [[1,2,3], [1,2], [3,4,7,9]]
        >>> pad_sequences_1d(test_data_list, dtype=np.float32)
        >>> test_data_3d = [np.random.randn(2,3,4), np.random.randn(4,3,4), np.random.randn(1,3,4)]
        >>> pad_sequences_1d(test_data_3d, dtype=np.float32)
    """
    if isinstance(sequences[0], list):
        if "torch" in str(dtype):
            sequences = [torch.tensor(s, dtype=dtype, device=device) for s in sequences]
        else:
            sequences = [np.asarray(s, dtype=dtype) for s in sequences]

    extra_dims = sequences[0].shape[1:]  # the extra dims should be the same for all elements
    lengths = [len(seq) for seq in sequences]
    if fixed_length is not None:
        max_length = fixed_length
    else:
        max_length = max(lengths)
    if isinstance(sequences[0], torch.Tensor):
        assert "torch" in str(dtype), "dtype and input type does not match"
        padded_seqs = torch.zeros((len(sequences), max_length) + extra_dims, dtype=dtype, device=device)
        mask = torch.zeros((len(sequences), max_length), dtype=torch.float32, device=device)
    else:  # np
        assert "numpy" in str(dtype), "dtype and input type does not match"
        padded_seqs = np.zeros((len(sequences), max_length) + extra_dims, dtype=dtype)
        mask = np.zeros((len(sequences), max_length), dtype=np.float32)

    for idx, seq in enumerate(sequences):
        end = lengths[idx]
        padded_seqs[idx, :end] = seq
        mask[idx, :end] = 1
    return padded_seqs, mask  # , lengths


def pad_sequences_2d(sequences, dtype=torch.long):
    """ Pad a double-nested list or a sequence of n-d torch tensor into a (n+1)-d tensor,
        only allow the first two dims has variable lengths
    Args:
        sequences: list(n-d tensor or list)
        dtype: torch.long for word indices / torch.float (float32) for other cases
    Returns:
    Examples:
        >>> test_data_list = [[[1, 3, 5], [3, 7, 4, 1]], [[98, 34, 11, 89, 90], [22], [34, 56]],]
        >>> pad_sequences_2d(test_data_list, dtype=torch.long)  # torch.Size([2, 3, 5])
        >>> test_data_3d = [torch.randn(2,2,4), torch.randn(4,3,4), torch.randn(1,5,4)]
        >>> pad_sequences_2d(test_data_3d, dtype=torch.float)  # torch.Size([2, 3, 5])
        >>> test_data_3d2 = [[torch.randn(2,4), ], [torch.randn(3,4), torch.randn(5,4)]]
        >>> pad_sequences_2d(test_data_3d2, dtype=torch.float)  # torch.Size([2, 3, 5])
    # TODO add support for numpy array
    """
    bsz = len(sequences)
    para_lengths = [len(seq) for seq in sequences]
    max_para_len = max(para_lengths)
    sen_lengths = [[len(word_seq) for word_seq in seq] for seq in sequences]
    max_sen_len = max([max(e) for e in sen_lengths])

    if isinstance(sequences[0], torch.Tensor):
        extra_dims = sequences[0].shape[2:]
    elif isinstance(sequences[0][0], torch.Tensor):
        extra_dims = sequences[0][0].shape[1:]
    else:
        sequences = [[torch.Tensor(word_seq, dtype=dtype) for word_seq in seq] for seq in sequences]
        extra_dims = ()

    padded_seqs = torch.zeros((bsz, max_para_len, max_sen_len) + extra_dims, dtype=dtype)
    mask = torch.zeros(bsz, max_para_len, max_sen_len).float()

    for b_i in range(bsz):
        for sen_i, sen_l in enumerate(sen_lengths[b_i]):
            padded_seqs[b_i, sen_i, :sen_l] = sequences[b_i][sen_i]
            mask[b_i, sen_i, :sen_l] = 1
    return padded_seqs, mask  # , sen_lengths
