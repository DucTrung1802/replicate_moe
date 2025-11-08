# Readme

## 1. Create virtual environment
Run following commands
```
python -m venv moe_env
.\moe_env\Scripts\activate
```

## 2. Install required libraries
```
pip install -r .\requirements.txt
```

## 3. Display instruction 
Run
```
cd .\cifar10\
python main.py -h
```
to know all arguments and their default values.

## 4. Examples
- For `MoE` architecture, model `resnet18`
    ```
    python main.py --model resnet18 --mixture
    ```

- For `Single` architecture, model `MobileNetV2`
    ```
    python main.py --model MobileNetV2 --no-mixture
    ```

- For `normal`, model `resnet18`
    ```
    python main.py --model resnet18
    ```

- For `normal`, model `MobileNetV2`
    ```
    python main.py --model MobileNetV2
    ```

- For `normal`, model `MobileNetV2`, max epoch `50`, early stopping with patience `5`, batch size `128`
    ```
    python main.py --model MobileNetV2 --max_epoch 50 --early_stop --patience 5 --batch_size 128
    ```
 

