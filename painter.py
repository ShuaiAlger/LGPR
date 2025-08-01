import numpy as np

from matplotlib import rcParams
import matplotlib.pyplot as plt
from scipy import interpolate

# plt.rc('font',family='Times New Roman')

params={
        # 'font.family':'serif',
        # 'font.serif':'Times New Roman',
        'font.weight':'normal', #or 'blod'
        'font.size':8.0,
        'pdf.fonttype': 42
        }
rcParams.update(params)



DPI = 500


from matplotlib import gridspec
# spec = gridspec.GridSpec(ncols=4, nrows=2, width_ratios=[7, 1, 7, 1])
spec = gridspec.GridSpec(ncols=6, nrows=3)

fig = plt.figure(figsize=(16, 10), dpi=DPI)






def read_txt(txt_file = ""):
    with open(txt_file) as f:
        lines = f.readlines()
        results = np.zeros((3, 40, 8))
        for i in range(0, 40):
            oneline = lines[65 + 13 * i].split("\n")[0].split("   ")
            results[0, i, 0] = float(oneline[4])
            results[0, i, 1] = float(oneline[5])
            results[0, i, 2] = float(oneline[6])
            results[0, i, 3] = float(oneline[7])
            results[0, i, 4] = float(oneline[8])
            results[0, i, 5] = float(oneline[9])
            results[0, i, 6] = float(oneline[10])
            results[0, i, 7] = float(oneline[11])

        for i in range(0, 40):
            oneline = lines[66 + 13 * i].split("\n")[0].split("   ")
            results[1, i, 0] = float(oneline[3])
            results[1, i, 1] = float(oneline[4])
            results[1, i, 2] = float(oneline[5])
            results[1, i, 3] = float(oneline[6])
            results[1, i, 4] = float(oneline[7])
            results[1, i, 5] = float(oneline[8])
            results[1, i, 6] = float(oneline[9])
            results[1, i, 7] = float(oneline[10])

        for i in range(0, 40):
            oneline = lines[67 + 13 * i].split("\n")[0].split("   ")
            results[2, i, 0] = float(oneline[2])
            results[2, i, 1] = float(oneline[3])
            results[2, i, 2] = float(oneline[4])
            results[2, i, 3] = float(oneline[5])
            results[2, i, 4] = float(oneline[6])
            results[2, i, 5] = float(oneline[7])
            results[2, i, 6] = float(oneline[8])
            results[2, i, 7] = float(oneline[9])

    results = results / 100.0

    return results





def upsample_area(thres, data1, data2):
    x1 = thres
    y1 = data1
    f = interpolate.interp1d(x1, y1, kind='cubic')
    nx1 = np.linspace(1, 10, 100)
    ny1 = f(nx1)

    x2 = thres
    y2 = data2
    f = interpolate.interp1d(x2, y2, kind='cubic')
    nx2 = np.linspace(1, 10, 100)
    ny2 = f(nx2)
    nx2 = np.flipud(nx2)
    ny2 = -0.02 + np.flipud(ny2)

    nx = np.concatenate([nx1, nx2])
    ny = np.concatenate([ny1, ny2])
    return nx, ny



def plot_curve():

    datasets = ["tjpark", "msls-eval", "pitts30k-eval"]

    methods = ["LGPR(sp_random)", "LGPR(sp_pretrain)", "LGPR(sp_freeze)",
               "LGPR(xf_random)", "LGPR(xf_pretrain)", "LGPR(xf_freeze)", "LGPR(xf_joint)"
               ]

    # metrics = ["R@1", "R@5", "R@10", "R@15", "R@20", "R@25", "R@50", "R@100"]
    metrics = ["R@1", "R@5", "R@10", "R@15", "R@20", "R@25"]

    all_results = []
    all_results.append(read_txt("./superpoint_random_trainall_outconv4a_gsvlight.txt"))
    all_results.append(read_txt("./superpoint_pretrain_trainall_outconv4a_gsvlight.txt"))
    all_results.append(read_txt("./superpoint_freeze_outputconv4a_boq_gsvlight.txt"))

    all_results.append(read_txt("./xfeat_random_output_x5sizedconcatx3x4x5_boq_gsvlight.txt"))
    all_results.append(read_txt("./xfeat_pretrain_output_x3sizeconcat_boq_gsvlight.txt"))
    all_results.append(read_txt("./xfeat_freeze_output_x3sizedconcatx3x4x5_boq_gsvlight.txt"))
    all_results.append(read_txt("./xfeat_gsv9000freeze_outputx3sized_boq_gsvlight.txt"))


    save_figs = 1
    if save_figs:
        lw = 1
        xcoords = np.arange(0, 40)
        colors = [  
                    'darkorchid',
                    'blue',
                    'orange',
                    'pink',
                    'mediumpurple', 
                    'green', 
                    'cyan', 

                    'pink', 
                    'lime',
                    ]

        linestyles = ['-', '-', '-', '--', '--', '--', '--', '--', '--', '--', '--', '--', '--', '--', '--']
        

        for d in range(len(datasets)):
            for j in range(len(metrics)):
                ax = fig.add_subplot(spec[j + len(metrics)*d])
                ax.set_xlim(1, 40)
                if j == 0:
                    ax.set_ylim(0, 1.0)
                else:
                    ax.set_ylim(0, 1.0)

                ax.set_xticks(range(0, 41, 10))

                for m in range(len(methods)):
                    ax.plot(xcoords, all_results[m][d, :, j], marker='', linestyle=linestyles[m], color=colors[m], linewidth=lw, label=methods[m])

                if j == 5:
                    ax.legend(bbox_to_anchor=(1.0, 1.0))
                if d == 0:
                    ax.set_title(metrics[j])

                ax.set_xlabel('Epoch')
                if j == 0:
                    ax.set_ylabel("Recall on " + datasets[d])

        plt.savefig('evaluation_curve.jpg', bbox_inches='tight', dpi=DPI)
        plt.savefig('evaluation_curve.pdf', bbox_inches='tight', dpi=DPI)
        plt.savefig('evaluation_curve.png', bbox_inches='tight', dpi=DPI)








if __name__ == '__main__':



    plot_curve()

