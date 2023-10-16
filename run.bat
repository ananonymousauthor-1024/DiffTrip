@echo off

REM Setting Python Interpreter Path(your python path)
set python_path=D:\anaconda3\envs\DiffTrip\python.exe

REM To Run T-Diff(DiffTrip)
REM The DiffTrip set default values T(step) in 32 and beta_t(max_noise) in 0.02, respectively. If you want to change it, try to add --step and --max_noise
python .\T-Diff\train_diffusion.py --dataset Osak --lr 0.01 --batch_size 4 --d_model 16

python .\T-Diff\train_diffusion.py --dataset Glas --lr 0.01 --batch_size 4 --d_model 64

python .\T-Diff\train_diffusion.py --dataset Edin --lr 0.01 --batch_size 16 --d_model 64

python .\T-Diff\train_diffusion.py --dataset Toro --lr 0.01 --batch_size 8 --d_model 64

REM To Run T-Diff-EPS(predict noise)
python .\T-Diff-EPS\train_diffusion.py --dataset Osak --lr 0.005 --batch_size 4 --d_model 32

python .\T-Diff-EPS\train_diffusion.py --dataset Glas --lr 0.005 --batch_size 4 --d_model 32

python .\T-Diff-EPS\train_diffusion.py --dataset Edin --lr 0.005 --batch_size 16 --d_model 32

python .\T-Diff-EPS\train_diffusion.py --dataset Toro --lr 0.005 --batch_size 8 --d_model 32

REM To Run T-Diff-FS(Fast Sampling)
python .\T-Diff-FS\train_diffusion.py --dataset Osak --lr 0.01 --batch_size 4 --d_model 16

python .\T-Diff-FS\train_diffusion.py --dataset Glas --lr 0.01 --batch_size 4 --d_model 64

python .\T-Diff-FS\train_diffusion.py --dataset Edin --lr 0.01 --batch_size 16 --d_model 64

python .\T-Diff-FS\train_diffusion.py --dataset Toro --lr 0.01 --batch_size 8 --d_model 64

REM To Run T-Base(Base)
REM max f1(pairs-f1) represent the results of Base-CM and total f1(pairs-f1) represent the results of Base
python .\T-Base\train_base.py --dataset Osak --lr 0.001 --batch_size 4 --d_model 128

python .\T-Base\train_base.py --dataset Glas --lr 0.001 --batch_size 4 --d_model 128

python .\T-Base\train_base.py --dataset Edin --lr 0.001 --batch_size 16 --d_model 128

python .\T-Base\train_base.py --dataset Toro --lr 0.001 --batch_size 8 --d_model 64

REM To Run T-Base-WSE(Weight of Start and End)
python .\T-Base-WSE\train_base.py --dataset Osak --lr 0.001 --batch_size 4 --d_model 128 --se_weight 5

python .\T-Base-WSE\train_base.py --dataset Glas --lr 0.001 --batch_size 4 --d_model 128 --se_weight 5

python .\T-Base-WSE\train_base.py --dataset Edin --lr 0.001 --batch_size 16 --d_model 128 --se_weight 5

python .\T-Base-WSE\train_base.py --dataset Toro --lr 0.001 --batch_size 8 --d_model 64 --se_weight 5






