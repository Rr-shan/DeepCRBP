import os
import statistics

from gp2lib import read_fasta_into_dic


def bpp_callback(v, v_size, i, maxsize, what, data=None):
    """
    This uses the Python3 API (RNA.py) of ViennaRNA (tested with v 2.4.13).
    So RNA.py needs to be in PYTHONPATH (it is if installed via conda).
    """
    import RNA
    if what & RNA.PROBS_WINDOW_BPP:
        data.extend([{'i': i, 'j': j, 'p': p} for j, p in enumerate(v) if (p is not None) and (p >= 0.01)])


def up_split_callback(v, v_size, i, maxsize, what, data):
    """
    This uses the Python3 API (RNA.py) of ViennaRNA (tested with v 2.4.13).
    So RNA.py needs to be in PYTHONPATH (it is if installed via conda).
    """
    import RNA
    if what & RNA.PROBS_WINDOW_UP:
        what = what & ~RNA.PROBS_WINDOW_UP
        dat = []
        # Non-split case:
        if what == RNA.ANY_LOOP:
            dat = data
        # all the cases where probability is split into different loop contexts
        elif what == RNA.EXT_LOOP:
            dat = data['ext']
        elif what == RNA.HP_LOOP:
            dat = data['hp']
        elif what == RNA.INT_LOOP:
            dat = data['int']
        elif what == RNA.MB_LOOP:
            dat = data['mb']
        dat.append({'i': i, 'up': v})


################################################################################
def calc_str_elem_up_bpp(in_fasta, out_bpp, out_str,
                         out_miss=False,
                         missing_ids_dic=None,
                         report=True,
                         stats_dic=None,
                         id2ucr_dic=False,
                         plfold_u=3,
                         plfold_l=100,
                         plfold_w=150):
    """
    Calculate structural elements probabilities (different loop contexts),
    as well as base pairs and their probabilities, using ViennaRNA.
    This uses the Python3 API (RNA.py) of ViennaRNA (tested with v 2.4.14).
    So RNA.py needs to be in PYTHONPATH, which it is,
    if e.g. installed via:
    conda install -c bioconda viennarna=2.4.14
    If no base pairs found for sequence, still print out ID header to
    out_bpp (just no base pair rows following).
    in_fasta:
        Input FASTA file
    out_bpp:
        Output base pair probabilities file 输出 碱基对概率 文件
    out_str:
        Output position-wise structural elements probabilities file
    out_miss:
        Output file to store FASTA IDs for which no BPs were found
    missing_ids_dic:
        Optionally, store missing IDs in missing_ids_dic. Like out_miss
        obsolete, since all sequence IDs are output even if no base pairs
        found for a given sequence.
    stats_dic:
        If not None, extract statistics from structure data and store
        in stats_dic.
    id2ucr_dic:
        Sequence ID to uppercase sequence start + end, with format:
        sequence_id -> "uppercase_start-uppercase_end"
        where both positions are 1-based.
        Set to define regions for which to generate element probability
        stats, stored in stats_dic.
    plfold_u:
        RNAplfold -u parameter value
    plfold_l:
        RNAplfold -L parameter value
    plfold_w:
        RNAplfold -W parameter value
    """
    # Import RNA.py library.
    try:
        import RNA
    except:
        assert False, "ViennaRNA Python3 API library RNA.py not in PYTHONPATH"
    # For real.ViennaRNa
    import RNA

    # Check input.
    assert os.path.isfile(in_fasta), "cannot open target FASTA file \"%s\"" % (in_fasta)
    # Read in FASTA file.
    seqs_dic = read_fasta_into_dic(in_fasta)
    # If stats dictionary given, compute statistics during run. 如果给出 stats 字典，则在运行期间计算统计信息。
    if stats_dic is not None:
        stats_dic["bp_c"] = 0
        stats_dic["seqlen_sum"] = 0
        stats_dic["nobpsites_c"] = 0
        stats_dic["seq_c"] = len(seqs_dic)
        pu_list = []
        ps_list = []
        pe_list = []
        ph_list = []
        pi_list = []
        pm_list = []
        pbp_list = []

    # Output files.
    OUTBPP = open(out_bpp, "w")
    OUTSTR = open(out_str, "w")
    if out_miss:
        OUTMISS = open(out_miss, "w")

    # Floor float, for centering probabilities (important when u > 1).
    i_add = int(plfold_u / 2)

    # Sequence counter.
    c_seq = 0

    # Calculate base pair and structural elements probabilities.
    if report:
        print("Calculate base pair and structural elements probabilities ... ")

    for seq_id, seq in sorted(seqs_dic.items()):

        md = RNA.md()
        md.max_bp_span = plfold_l
        md.window_size = plfold_w

        # Get base pairs and their probabilities.
        data = []
        # Different loop context probabilities.
        data_split = {'ext': [], 'hp': [], 'int': [], 'mb': []}

        fc = RNA.fold_compound(seq, md, RNA.OPTION_WINDOW)
        # Get base pairs and their probabilities.
        fc.probs_window(plfold_u, RNA.PROBS_WINDOW_BPP, bpp_callback, data)
        # Get different loop context probabilities.
        fc.probs_window(plfold_u, RNA.PROBS_WINDOW_UP | RNA.PROBS_WINDOW_UP_SPLIT, up_split_callback, data_split)

        # If base pairs found.
        if data:
            # print(data)
            # Output base pair probabilities.
            OUTBPP.write(">%s\n" % (seq_id))
            for prob in data:
                p = prob['p']
                i = prob['i']
                j = prob['j']
                OUTBPP.write("%i\t%i\t%f\n" % (i, j, p))
                if stats_dic:
                    stats_dic["bp_c"] += 1
                    pbp_list.append(p)
        else:
            if report:
                print("WARNING: no base pairs found for \"%s\"" % (seq_id))
            # Still print header.
            OUTBPP.write(">%s\n" % (seq_id))
            if stats_dic:
                stats_dic["nobpsites_c"] += 1
            if out_miss:
                OUTMISS.write("%s\n" % (seq_id))
            if missing_ids_dic is not None:
                missing_ids_dic[seq_id] = 1

        # Store individual probs for sequence in lists.
        ups = []
        ups_e = []
        ups_h = []
        ups_i = []
        ups_m = []
        ups_s = []

        for i, e in enumerate(seq):
            data_i = i + 1
            p_e = 0
            p_h = 0
            p_i = 0
            p_m = 0
            if data_split['ext'][i]['up'][plfold_u]:
                p_e = data_split['ext'][i]['up'][plfold_u]
            if data_split['hp'][i]['up'][plfold_u]:
                p_h = data_split['hp'][i]['up'][plfold_u]
            if data_split['int'][i]['up'][plfold_u]:
                p_i = data_split['int'][i]['up'][plfold_u]
            if data_split['mb'][i]['up'][plfold_u]:
                p_m = data_split['mb'][i]['up'][plfold_u]
            # Total unpaired prob = sum of different loop context probs.
            p_u = p_e + p_h + p_i + p_m
            if p_u > 1:
                p_u = 1
            # Paired prob (stacked prob).
            p_s = 1 - p_u
            ups.append(p_u)
            ups_e.append(p_e)
            ups_h.append(p_h)
            ups_i.append(p_i)
            ups_m.append(p_m)
            ups_s.append(p_s)

        # Center the values and output for each sequence position.
        OUTSTR.write(">%s\n" % (seq_id))
        l_seq = len(seq)
        if stats_dic is not None:
            stats_dic["seqlen_sum"] += l_seq
        for i, c in enumerate(seq):
            # At start, end, and middle.
            if i < i_add:
                p_u = ups[plfold_u - 1]
                p_e = ups_e[plfold_u - 1]
                p_h = ups_h[plfold_u - 1]
                p_i = ups_i[plfold_u - 1]
                p_m = ups_m[plfold_u - 1]
                p_s = ups_s[plfold_u - 1]
            elif i >= (l_seq - i_add):
                p_u = ups[l_seq - 1]
                p_e = ups_e[l_seq - 1]
                p_h = ups_h[l_seq - 1]
                p_i = ups_i[l_seq - 1]
                p_m = ups_m[l_seq - 1]
                p_s = ups_s[l_seq - 1]
            else:
                p_u = ups[i + i_add]
                p_e = ups_e[i + i_add]
                p_h = ups_h[i + i_add]
                p_i = ups_i[i + i_add]
                p_m = ups_m[i + i_add]
                p_s = ups_s[i + i_add]
            # Output centered values.
            pos = i + 1  # one-based sequence position.
            OUTSTR.write("%i\t%f\t%f\t%f\t%f\t%f\t%f\n" % (pos, p_u, p_e, p_h, p_i, p_m, p_s))
            OUTSTR.write("%f\t%f\t%f\t%f\t%f\n" % (p_e, p_h, p_i, p_m, p_s))
            if stats_dic:
                if id2ucr_dic:
                    # If id2ucr_dic, record values only for uppercase part of sequence.
                    uc_s = id2ucr_dic[seq_id][0]
                    uc_e = id2ucr_dic[seq_id][1]
                    if pos >= uc_s and pos <= uc_e:
                        pu_list.append(p_u)
                        ps_list.append(p_s)
                        pe_list.append(p_e)
                        ph_list.append(p_h)
                        pi_list.append(p_i)
                        pm_list.append(p_m)
                else:
                    pu_list.append(p_u)
                    ps_list.append(p_s)
                    pe_list.append(p_e)
                    ph_list.append(p_h)
                    pi_list.append(p_i)
                    pm_list.append(p_m)

        c_seq += 1
        if report:
            if not c_seq % 100:
                print("%i sequences processed" % (c_seq))

    OUTBPP.close()
    OUTSTR.close()
    if out_miss:
        OUTMISS.close()

    # Calculate stats if stats_dic set.
    if stats_dic:
        # Mean values.
        stats_dic["U"] = [statistics.mean(pu_list)]
        stats_dic["S"] = [statistics.mean(ps_list)]
        stats_dic["E"] = [statistics.mean(pe_list)]
        stats_dic["H"] = [statistics.mean(ph_list)]
        stats_dic["I"] = [statistics.mean(pi_list)]
        stats_dic["M"] = [statistics.mean(pm_list)]
        stats_dic["bp_p"] = [statistics.mean(pbp_list)]
        # Standard deviations.
        stats_dic["U"] += [statistics.stdev(pu_list)]
        stats_dic["S"] += [statistics.stdev(ps_list)]
        stats_dic["E"] += [statistics.stdev(pe_list)]
        stats_dic["H"] += [statistics.stdev(ph_list)]
        stats_dic["I"] += [statistics.stdev(pi_list)]
        stats_dic["M"] += [statistics.stdev(pm_list)]
        stats_dic["bp_p"] += [statistics.stdev(pbp_list)]


'''
names = ['10_PARCLIP_ELAVL1A_hg19', '11_CLIPSEQ_ELAVL1_hg19', '12_PARCLIP_EWSR1_hg19', '13_PARCLIP_FUS_hg19',
         '14_PARCLIP_FUS_mut_hg19', '15_PARCLIP_IGF2BP123_hg19', '16_ICLIP_hnRNPC_Hela_iCLIP_all_clusters',
         '17_ICLIP_HNRNPC_hg19',
         '18_ICLIP_hnRNPL_Hela_group_3975_all-hnRNPL-Hela-hg19_sum_G_hg19--ensembl59_from_2337-2339-741_bedGraph-cDNA-hits-in-genome',
         '19_ICLIP_hnRNPL_U266_group_3986_all-hnRNPL-U266-hg19_sum_G_hg19--ensembl59_from_2485_bedGraph-cDNA-hits-in-genome',
         '1_PARCLIP_AGO1234_hg19',
         '20_ICLIP_hnRNPlike_U266_group_4000_all-hnRNPLlike-U266-hg19_sum_G_hg19--ensembl59_from_2342-2486_bedGraph-cDNA-hits-in-genome',
         '21_PARCLIP_MOV10_Sievers_hg19',
         '22_ICLIP_NSUN2_293_group_4007_all-NSUN2-293-hg19_sum_G_hg19--ensembl59_from_3137-3202_bedGraph-cDNA-hits-in-genome',
         '23_PARCLIP_PUM2_hg19', '24_PARCLIP_QKI_hg19', '25_CLIPSEQ_SFRS1_hg19', '26_PARCLIP_TAF15_hg19',
         '27_ICLIP_TDP43_hg19',
         '28_ICLIP_TIA1_hg19', '29_ICLIP_TIAL1_hg19', '2_PARCLIP_AGO2MNASE_hg19',
         '30_ICLIP_U2AF65_Hela_iCLIP_ctrl_all_clusters', '31_ICLIP_U2AF65_Hela_iCLIP_ctrl+kd_all_clusters',
         '3_HITSCLIP_Ago2_binding_clusters', '4_HITSCLIP_Ago2_binding_clusters_2', '5_CLIPSEQ_AGO2_hg19',
         '6_CLIP-seq-eIF4AIII_1', '7_CLIP-seq-eIF4AIII_2', '8_PARCLIP_ELAVL1_hg19', '9_PARCLIP_ELAVL1MNASE_hg19']

for name in names:
    # in_fasta = './linkdataset/' + name + '/negative.fa'
    in_fasta = './linkdataset/' + name + '_train/negative'
    out_bpp = './linkdataset/' + name + '_train/negative.bpp.str'
    out_str = './linkdataset/' + name + '_train/negative.elem_p.str'
    pos_str_stats_dic = {}
    # print(name)  
    # negative positive

    calc_str_elem_up_bpp(in_fasta, out_bpp, out_str,
                         stats_dic=pos_str_stats_dic,
                         plfold_u=3,
                         plfold_l=100,
                         plfold_w=150)
    # print(pos_str_stats_dic)
    # print(len(pos_str_stats_dic))
'''
