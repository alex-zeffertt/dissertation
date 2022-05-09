# Run ./model_complete.py with every different combination of inputs and build csv output
import numpy as np
from subprocess import check_output

csv_columns = [
    'mean_n_weak_ties','modal_weak_tie_km','awareness_pc','facility_pc','p_update_logit_normal_sigma',
    'percent_no_ties','percent_only_weak_ties','percent_1_to_2_strong_no_weak_ties', 'percent_1_to_2_strong_plus_weak_ties','percent_3_or_more_strong_ties',
    'percent_no_intention','percent_intention','percent_reducer']
print(', '.join(map(lambda s:f'"{s}"', csv_columns)))

for mean_n_weak_ties in range(2,9):
    for modal_weak_tie_km in range(1,6):
        for awareness_pc in range(10,100,20):
            for facility_pc in range(10,100,20):
                for p_update_logit_normal_sigma in np.arange(.25,3,.5):
                    cmd = './model_complete.py '\
                        f'--mean_n_weak_ties={mean_n_weak_ties} '\
                        f'--modal_weak_tie_km={modal_weak_tie_km} '\
                        f'--awareness_pc={awareness_pc} '\
                        f'--facility_pc={facility_pc} '\
                        f'--p_update_logit_normal_sigma={p_update_logit_normal_sigma} '
                    output = check_output(cmd, shell=True)
                    lines=output.decode().split('\n')
                    X = eval(lines[0].split('=')[1])
                    Y = eval(lines[1].split('=')[1])
                    # convert to %
                    X = (100*np.array(X)/sum(X)).tolist()
                    Y = (100*np.array(Y)/sum(Y)).tolist()
                    csv_output_line = [mean_n_weak_ties,modal_weak_tie_km,awareness_pc,facility_pc,p_update_logit_normal_sigma] + X + Y
                    print(', '.join(map(str,csv_output_line)))
