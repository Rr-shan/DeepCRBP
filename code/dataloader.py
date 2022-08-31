
from gp2lib import read_bpp_into_dic, extract_uc_region_coords_from_fasta, read_fasta_into_dic, read_feat_into_dic, \
    string_vectorizer
import re
import torch
from torch_geometric.data import DataLoader, Data
from Deal_Kmer import *


def dataload(protein):
    print(protein)
    emb = np.load('../emb_npy/embedding.npy')
    pro = np.load('../emb_npy/' + protein + '.npy')
    pro_this = emb[pro]

    Kmer = dealwithdata(protein)

    pos_bpp_in = "../dataset/" + protein + "/positive.bpp.str"
    neg_bpp_in = "../dataset/" + protein + "/negative.bpp.str"
    pos_fa_in = "../dataset/" + protein + "/positive.fa"
    nes_fa_in = "../dataset/" + protein + "/negative.fa"

    bps_mode = 1

    seqs_dic = {}
    seqs_dic = read_fasta_into_dic(pos_fa_in,
                                   all_uc=True)  # seqs_dic = read_fasta_into_dic(pos_fa_in, all_uc=seqs_all_uc)
    pos_ids_dic = {}
    for seq_id in seqs_dic:
        pos_ids_dic[seq_id] = 1

    seqs_dic = read_fasta_into_dic(nes_fa_in, all_uc=True, seqs_dic=seqs_dic)
    neg_ids_dic = {}
    for seq_id in seqs_dic:
        if seq_id not in pos_ids_dic:
            neg_ids_dic[seq_id] = 1

    vp_dic = extract_uc_region_coords_from_fasta(seqs_dic)

    bpp_dic = read_bpp_into_dic(pos_bpp_in, vp_dic,
                                bps_mode=bps_mode)
    bpp_dic = read_bpp_into_dic(neg_bpp_in, vp_dic,
                                bpp_dic=bpp_dic,
                                bps_mode=bps_mode)

    label_list = []
    seq_ids_list = []
    for seq_id, c in sorted(pos_ids_dic.items()):
        seq_ids_list.append(seq_id)
        label_list.append(1)

    for seq_id, c in sorted(neg_ids_dic.items()):
        seq_ids_list.append(seq_id)
        label_list.append(0)

    feat_dic = {}

    for seq_id in seqs_dic:
        seq = seqs_dic[seq_id]
        # feat_dic[seq_id] = string_vectorizer(seq, custom_alphabet=fid2cat_dic["fa"])
        feat_dic[seq_id] = string_vectorizer(seq)

    undirected = 0

    coden_dict = {'GCU': 0, 'GCC': 0, 'GCA': 0, 'GCG': 0,  # alanine<A>
                  'UGU': 1, 'UGC': 1,  # systeine<C>
                  'GAU': 2, 'GAC': 2,  # aspartic acid<D>
                  'GAA': 3, 'GAG': 3,  # glutamic acid<E>
                  'UUU': 4, 'UUC': 4,  # phenylanaline<F>
                  'GGU': 5, 'GGC': 5, 'GGA': 5, 'GGG': 5,  # glycine<G>
                  'CAU': 6, 'CAC': 6,  # histidine<H>
                  'AUU': 7, 'AUC': 7, 'AUA': 7,  # isoleucine<I>
                  'AAA': 8, 'AAG': 8,  # lycine<K>
                  'UUA': 9, 'UUG': 9, 'CUU': 9, 'CUC': 9, 'CUA': 9, 'CUG': 9,  # leucine<L>
                  'AUG': 10,  # methionine<M>
                  'AAU': 11, 'AAC': 11,  # asparagine<N>
                  'CCU': 12, 'CCC': 12, 'CCA': 12, 'CCG': 12,  # proline<P>
                  'CAA': 13, 'CAG': 13,  # glutamine<Q>
                  'CGU': 14, 'CGC': 14, 'CGA': 14, 'CGG': 14, 'AGA': 14, 'AGG': 14,  # arginine<R>
                  'UCU': 15, 'UCC': 15, 'UCA': 15, 'UCG': 15, 'AGU': 15, 'AGC': 15,  # serine<S>
                  'ACU': 16, 'ACC': 16, 'ACA': 16, 'ACG': 16,  # threonine<T>
                  'GUU': 17, 'GUC': 17, 'GUA': 17, 'GUG': 17,  # valine<V>
                  'UGG': 18,  # tryptophan<W>
                  'UAU': 19, 'UAC': 19,  # tyrosine(Y)
                  'UAA': 20, 'UAG': 20, 'UGA': 20,  # STOP code
                  }

    # the amino acid code adapting 21-dimensional vector (20 amino acid and 1 STOP code)

    def coden(seq):
        vectors = np.zeros((len(seq) - 2, 21))
        for i in range(len(seq) - 2):
            vectors[i][coden_dict[seq[i:i + 3].replace('T', 'U')]] = 1
        return vectors.tolist()  # 21X99

    all_graphs = []

    for idx, label in enumerate(label_list):
        seq_id = seq_ids_list[idx]
        seq = seqs_dic[seq_id]  # AUGC
        l_seq = len(seq)  # 101

        # Make edge indices for backbone.
        edge_index_1 = []
        edge_index_2 = []
        # Edge indices 0-based!
        for n_idx in range(l_seq - 1):
            edge_index_1.append(n_idx)
            edge_index_2.append(n_idx + 1)
            # In case of undirected graphs, add backward edges too.
            if undirected:
                edge_index_1.append(n_idx + 1)
                edge_index_2.append(n_idx)

        # Add base pair edges.
        if bpp_dic:
            vp_s = vp_dic[seq_id][0]
            # print('vp_s', vp_s)
            vp_e = vp_dic[seq_id][1]
            # print('vp_e', vp_e)

            # Entry e.g. 'CLIP_01': ['130-150,0.33', '160-200,0.44', '240-260,0.55']
            for entry in bpp_dic[seq_id]:
                m = re.search("(\d+)-(\d+),(.+)", entry)
                p1 = int(m.group(1))  # 1-based.
                p2 = int(m.group(2))  # 1-based.
                bpp_value = float(m.group(3))

                g_p1 = p1 - 1  # 0-based base pair index.
                g_p2 = p2 - 1  # 0-based base pair index.
                # Filter.
                # if bpp_value < args.bps_cutoff: continue
                if bpp_value < 0.5: continue
                # Add base pair depending on set mode.
                add_edge = False
                if bps_mode == 1:
                    if (p1 >= vp_s and p1 <= vp_e) or (p2 >= vp_s and p2 <= vp_e):
                        add_edge = True
                elif bps_mode == 2:
                    if p1 >= vp_s and p2 <= vp_e:
                        add_edge = True
                if add_edge:
                    edge_index_1.append(g_p1)
                    edge_index_2.append(g_p2)
                    if undirected:
                        edge_index_1.append(g_p2)
                        edge_index_2.append(g_p1)

        # Merge edge indices.
        edge_index = torch.tensor([edge_index_1, edge_index_2], dtype=torch.long)
        x = torch.tensor(feat_dic[seq_id], dtype=torch.float)
        # print('edge_index',edge_index)
        # data = Data(x=x, edge_index=edge_index.t().contiguous(), y=label)

        data = Data(x=x, edge_index=edge_index, y=label)

        target = pro_this[idx].tolist()
        # target = pro_this[idx]
        data.target = torch.FloatTensor([target]).transpose(1, 2)
        # target01 = coden(seq)
        target01 = Kmer[idx].tolist()
        data.target01 = torch.FloatTensor([target01]).transpose(1, 2)

        all_graphs.append(data)
    return all_graphs
