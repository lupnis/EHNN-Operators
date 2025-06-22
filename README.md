嵌入超网络nn算子

被横向fully occupied了，不想搞这个和那些adaptive相关的研究了，故此直接开源

如果有帮助求个star

当时（2024-9-25或者更早，拿实验室空闲机器偷偷跑ai画图玩，想到此idea然后和C老师、R老师聊了下）ai绘图中的prompt：如果我要画一个动物，它有0.1像猫，0.9像狗，在传统架构里面，权重不变的时候，0.1、0.9并不会比0.5猫/狗更引起注意力的变化（搞了个小的可视化看了下），相比之下都没随机种子的影响大… 所以对于不同的输入表征，其实应该有个adaptation的过程：通过两阶段，基于输入（或者其特征）对模型权重进行注入。

同理，该方法可以用于适应不通任务的过程，通过调整injection，达到adaptive或者类似moe的效果。

目前里面有个超参，控制injection力度，可能可以改进（？）

训练过程：训练一个base，freeze掉它，然后训练权重注入部分。

Embedded Hypernetwork NN Operator

It’s been fully occupied laterally, and I don't want to work on this or those adaptation-related studies anymore, so I'm open-sourcing it directly.

If it helps you, please consider giving it a star.

Back then (around 2024-09-25 or earlier, I was secretly using idle lab machines to play around with AI-generated images), I came up with this idea while thinking about prompts in AI drawing:
If I want to draw an animal that is 0.1 like a cat and 0.9 like a dog, in traditional architectures, with fixed weights, 0.1/0.9 doesn’t produce more attention change than a 0.5 cat/dog prompt (I did some simple visualizations to verify this). The differences were even less impactful than the random seed...

So, for different input representations, there should actually be an adaptation process: a two-stage approach that injects weights into the model based on the input (or its features).

Similarly, this method can be used to adapt to different tasks by adjusting the injection, achieving an adaptive effect or something similar to a Mixture-of-Experts (MoE) approach.

Currently, there’s a hyperparameter inside that controls the strength of the injection — might be improvable (?)

Training process: Train a base model, freeze it, then train the weight injection component.
