#!binbash

# SDDMM
python sddmm_fp16_test_args.py 32 &&

python sddmm_fp16_test_args.py 128 &&

python sddmm_tf32_test_args.py 32 &&

python sddmm_tf32_test_args.py 128
        