import os
import json
import shutil
import sys
import torch
import logging
def setup_exp(opt, copy_dirs: list=[]):
    """
    Creates directories, copies main file to it and saves the configuration
    attribute: gpu then it will set it set it as default cuda device for torch.
    attribute: model_file then it will create the directory and save the opt
               object to json and copy the file to that directory
    :param opt: Output from parser.parse_args()
    :param copy_dirs: list of directories to optionally copy to experiment
                      directory
    """
    log = logging.getLogger('rutil')
    if hasattr(opt, 'gpu') and torch.cuda.is_available():
        # TODO: add tensorflow support?
        # If str then csv parse
        if isinstance(opt.gpu, str):
            opt.gpu = [int(x) for x in opt.gpu.split(",")]
            torch.cuda.set_device(opt.gpu[0])
        elif isinstance(opt.gpu, list):
            torch.cuda.set_device(opt.gpu[0])
        elif isinstance(opt.gpu, int):
            opt.gpu = opt.gpu
            torch.cuda.set_device(opt.gpu)
        else:
            raise ValueError(f"Unknown GPU datatype passed {opt.gpu}, should be"
                             " either csv string, int or list of ints")
        log.info("Setting GPU Device to {}".format(opt.gpu))
    # Don't save
    if opt.model_file is None:
        return
    # creating output folders
    if not os.path.exists(opt.model_file):
        os.makedirs(opt.model_file)
    # Copy file to result directory for reproducability
    _fname = os.path.basename(sys.argv[0])
    main_file = os.path.realpath(_fname)
    cp_filename = os.path.join(opt.model_file, _fname)
    if not os.path.exists(cp_filename):
        log.info(f"Copying {main_file} ==> {cp_filename}")
        # print(f"Copying {main_file} ==> {cp_filename}")
        shutil.copy2(main_file, cp_filename)
        with open(os.path.join(opt.model_file, "config.json"), 'w') as f:
            json.dump(opt.__dict__, f, sort_keys=True, indent=2)
        # Recursively copy directories
        for d in copy_dirs:
            d_path = os.path.join(opt.model_file, d)
            log.info(f"Copying {d} ==> {d_path}")
            shutil.copytree(d, d_path,
                            ignore=shutil.ignore_patterns('*.pyc', '.*', '#*',
                                                          '*.mtx', '*.pkl',
                                                          '__pycache__'))