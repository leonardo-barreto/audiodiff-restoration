import os
import hydra
import torch
import torchaudio

def _main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    global __file__
    __file__ = hydra.utils.to_absolute_path(__file__)
    dirname = os.path.dirname(__file__)
    args.model_dir = os.path.join(dirname, str(args.model_dir))
    if not os.path.exists(args.model_dir):
        raise FileNotFoundError(f"Model directory {args.model_dir} does not exist")
    
    torch.cuda.empty_cache()
    print("Starting run ...")
    print(f"Noisy file: {args.noisy_audio}")
    print(f"Output file: {args.output_file}")
    print(f"Device: {device}")
    print(f"Noise floor window size: {args.inference.denoising.power_window_size}")
    #print(f"Xi: {args.inference.xi}\n\n\n")

    args.exp.model_dir = args.model_dir

    #################
    ## Diff params ##
    #################

    diff_params = hydra.utils.instantiate(args.diff_params)

    #############
    ## Network ##
    #############

    # it prints some logs.
    network = hydra.utils.instantiate(args.network, device=device)
    network = network.to(device)

    #############
    ## Sampler ##
    #############
    #from sampling.Sampler78rpm import Sampler78rpm
    #sampler = Sampler78rpm(args, network, diff_params)

    #################
    ## Conditioner ##
    #################

    ############
    ## Tester ##
    ############

    if args.tester.tester._target_ == "testing.tester78rpm.Tester78rpm":
        from testing.tester78rpm import Tester78rpm
    else:
        raise ValueError(f"tester target {args.tester.tester._target_} not recognized")
    
    tester = Tester78rpm(args=args, network=network, diff_params=diff_params, device=device)

    # Loading checkpoint
    if str(args.checkpoint) != 'None':
        try:
            # relative path
            tester.load_checkpoint(os.path.join(dirname, args.checkpoint))
        except FileNotFoundError:
            print('Checkpoint not found in relative path')
            pass
            try:
                # modeldir path
                tester.load_checkpoint(os.path.join(args.model_dir, args.checkpoint))
            except FileNotFoundError:
                # absolute path
                print('Checkpoint not found in modeldir path')
                tester.load_checkpoint(args.checkpoint)
    else:
        raise ValueError("Checkpoint path is None. Please provide a valid checkpoint path in the config file.")

    # Setup noisy audio blocks and output length
    tester.setup_data()
    if args.debug == True:
        print(f"Noisy audio setup")
        print(f"Output length: {tester.out_len}")
        print(f"Number of blocks: {len(tester.noisy_audio_blocks.dataset)}")
        print(f"Blocks' shape: {tester.noisy_audio_blocks.dataset.blocks.shape}")

    # Perform inference
    denoised = tester.inference()
    if args.debug == True:
        print(f"Inference successful")
        print(f"Output shape: {denoised.shape}")

    # Save the denoised audio to the output file
    torchaudio.save(
        args.output_file,
        denoised,
        args.sample_rate,
        bits_per_sample=16,
        encoding="PCM_S",
    )

#-----------------------------------------------------------------------------------------

@hydra.main(config_path="conf", config_name="conf") #, version_base=str(hydra.__version__))
def main(args):
    try:
        torch.cuda.set_device(args.gpu)
    except Exception:
        pass
    _main(args)


if __name__ == "__main__":
    main()


