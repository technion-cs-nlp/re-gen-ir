import json
import os
from collections import defaultdict 
from argparse import ArgumentParser
from tqdm import tqdm


parser = ArgumentParser(prog='Summary Neuron Inputs')
parser.add_argument('-p', '--path', help='path of the all_queries_neuron_similarities files to evaluate')
parser.add_argument('--sim-type', help='cosine or dot')
parser.add_argument('--topk', help='top k conponents to average', default=None)



if __name__ == "__main__":
    args = parser.parse_args()


    layer_head_to_all_top_k_components = defaultdict(list)
    k = args.topk

    resid_pres = [f'decoder.{l}.hook_resid_pre' for l in range(1,24)]
    resid_post = [f'decoder.{l}.hook_resid_post' for l in range(24)]
    resid_mid = [f'decoder.{l}.hook_resid_mid_cross' for l in range(24)]
    heads = [f'decoder.{l}.{comp}.head{h}' for l in range(24) for comp in ['cross_attn', 'attn'] for h in range(16)]
    remove_comp = resid_pres + resid_post + resid_mid + heads

    path = args.path
    file_name_start = 'all_queries_neuron_similarities_'+ args.sim_type
    files = [f for f in os.listdir(path) if f.startswith(file_name_start)]


    for f in tqdm(files):
        for entry in json.load(open(f"{path}/{f}")):
            layer = entry['layer']

            position = entry['position']
            component_list = entry['value']

            filtered_sims = [(component, component_list[component]) for component in component_list if component not in remove_comp and component_list[component] > 0]
            sorted_sims = list(sorted(filtered_sims, key=lambda x: x[1], reverse=True))
            if k:
                layer_head_to_all_top_k_components[(position, layer)].append(sorted_sims[:int(k)])
            else:
                layer_head_to_all_top_k_components[(position, layer)].append(sorted_sims)

    def to_ratio(values):
        if len(values) == 0:
            return values
        comps, comp_vals = zip(*values)

        val_sum = sum(comp_vals)
        ratios = [v/val_sum for v in comp_vals]
        return list(zip(comps, ratios))

    def average_lists(lists):

        sum_components = defaultdict(float)
        
        for l in lists:
            for i in range(len(l)):
                comp = l[i][0]
                sum_components[comp] += l[i][1]

        

        avgs =  {comp: sum_components[comp] / len(lists) for comp in sum_components}
        return [(comp, avgs[comp]) for comp in avgs]

    ratio_of_top_k = {}
    layer_to_positions = defaultdict(list)
    for key in layer_head_to_all_top_k_components:
        pos, layer = key
        layer_to_positions[layer].append(pos)
        list_of_comps = layer_head_to_all_top_k_components[key]
        list_of_percents = [to_ratio(c) for c in list_of_comps]
        averages = average_lists(list_of_percents)
        ratio_of_top_k[key] = averages

    ratio_of_top_k_layers = {}
    for layer in layer_to_positions:
        list_of_comps = []
        for pos in layer_to_positions[layer]:
            key = (pos, layer)
            list_of_comps.append(ratio_of_top_k[key])
        list_of_percents = [to_ratio(c) for c in list_of_comps]
        averages = average_lists(list_of_percents)
        ratio_of_top_k_layers[layer] = averages

    k = 'all' if not k else k
    json.dump(ratio_of_top_k_layers, open(f'{path}/neuron_ratio_of_top_k_layers-{args.sim_type}-{k}.json', 'w'))
