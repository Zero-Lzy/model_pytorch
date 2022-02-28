# 本模型为15年Hinton知识蒸馏开山之作的复现
--torch 1.7.1--

模型在model.py中

teacher.py先训练teacher model

再通过distill.py对teacher model蒸馏

