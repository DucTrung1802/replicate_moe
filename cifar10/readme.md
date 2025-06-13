# Readme

## 1. Display instruction 
Run `python main.py -h` to know all arguments and their default values.

## 2. Examples
- For `MoE`, model `resnet18`
    ```
    python main.py --model resnet18 --mixture
    ```

- For `MoE`, model `MobileNetV2`
    ```
    python main.py --model MobileNetV2 --mixture
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
 

