import argparse
import numpy as np
import ROOT
import root_numpy as rnp
import h5py

import utils
import os
import json
from collections import OrderedDict 



def calc_normalization(path, tree, branches):
    tfile = ROOT.TFile(path, 'READ')
    tree = tfile.Get(tree)
    mean = np.zeros(len(branches))
    std = np.ones(len(branches))
    for i, branch in enumerate(branches):
        arr = rnp.tree2array(tree, branches=branch)
        mean[i] = np.mean(arr)
        std[i] = np.std(arr)
        if std[i] == 0:
            std[i] = 1
    return {'std': std, 'mean': mean}


def save_json(branches, norm):
    # Prepare for saving information to lwtnn
    # See https://github.com/dguest/lwtnn-tutorial-for-btaggers for motivation for this section
    variables_json = OrderedDict([('inputs', []),
                      ('outputs', [])])

    input_variables = OrderedDict([("name", "NNinputs"),("variables", [])])
    for i,var_name in enumerate(branches[0]):
        #variable_dict = {'name': var_name, 'offset': norm["mean"][i], 'scale': 1.0/norm["std"][i] }
        variable_dict = OrderedDict([('name', var_name), ('offset', norm["mean"][i]), ('scale', 1.0/norm["std"][i])])
        input_variables["variables"].append(variable_dict)
    variables_json['inputs'].append(input_variables)

    #Variables of one Gausian Mixture Model (GMM) unit
    GMM_vars = [
            "alpha",  #alpha = mixing coefficient
            "mean_x",
            "mean_y",
            "prec_x",
            "prec_y"]

    #Write the output variables
    output_variables = OrderedDict([('outputs', [])])
    for j in range (int(len(branches[1])/2)):
        outfile = 'variables_pos%s'%(str(j+1))
        GMM_unit = OrderedDict()
        GMM_unit["labels"] = GMM_vars
        GMM_unit["name"] = "merge_%s"%(str(j+1))
        variables_json["outputs"].append(GMM_unit)


    # now we will save the output files(create a subdirectory to keep it clean)
    model_dir = 'MDN_variables'
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)
    print()
    with open('{}/{}.json'.format(model_dir, outfile),'w') as variables:
        variables.write(json.dumps(variables_json, indent=2))
    print("variables.json written to {}".format(model_dir))


def get_MDNinputs(training_input,
             validation_fraction,
             output,
             config): \
            # pylint: disable=too-many-arguments,too-many-locals,dangerous-default-value
    
    """ train a neural network
    arguments:
    training_input -- path to ROOT training dataset
    validation_fraction -- held-out fraction of dataset for early-stopping
    output -- prefix for output files
    """

    branches = utils.get_data_config_names(config, meta=False)
    print("Branches are:   ")
    print(branches)
    norm = calc_normalization(
        training_input,
        'NNinput',
        branches[0]
    )
    json = save_json(branches,norm)
    print(json)


def _main():

    parse = argparse.ArgumentParser()
    parse.add_argument('--training-input', required=True)
    parse.add_argument('--output', required=True)
    parse.add_argument('--config', required=True)
    parse.add_argument('--validation-fraction', type=float, default=0.1)  #no valid set included
    args = parse.parse_args()

    get_MDNinputs(
        args.training_input,
        args.validation_fraction,
        args.output,
        args.config
    )



if __name__ == '__main__':
    _main()
