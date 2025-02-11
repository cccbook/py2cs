以下是使用散度與旋度描述的馬克斯威方程式。

定律 			| 微觀公式 (使用散度、旋度)   | 巨觀公式 (使用通量、環量) | 說明
----------------------|---------------------------|------------------------|------
法拉第定律 	| $\nabla \times \mathbf{E} = -\frac{\partial \mathbf{B}} {\partial t}$ | $\oint_{\mathbb{L}}\ \mathbf{E} \cdot \vec{dl}  = - \frac {\mathrm{d} \Phi_\mathbf{B}}{\mathrm{d} t}$ | 磁通量 B 的變化會產生感應電場 E
安培定律 		| $\nabla \times \mathbf{H} = \mathbf{J} + \frac{\partial \mathbf{D}} {\partial t}$ | $\oint_{\mathbb{L}}\ \mathbf{H} \cdot \vec{dl} = I_{f} + \frac {\mathrm{d} \Phi_\mathbf{D}}{\mathrm{d} t}$ | 電流 J 與電通量變化 $\frac{\partial \mathbf{D}} {\partial t}$  會產生磁場 H
高斯定律 		| $\nabla \cdot \mathbf{D} = \rho$ | $\oint_{S} D\cdot\vec{ds} = Q_{f}$ | 電荷密度 $\rho$ 決定電通量 D
自然定律 		| $\nabla \cdot \mathbf{B} = 0$ | $\oint_{S} B\cdot\vec{ds} = 0$ | 進入任一區域的磁通量一定等於出去的磁通量

如果是在相同的介質當中，上述方程式裏的介電率 $\epsilon$ 與導磁率 $\mu$ 就會是固定的，此時整個馬克斯威方程式就可以進一步簡化為下列兩條：

定律 			| 公式 					| 說明
----------------------|---------------------------------------|------
法拉第定律 	| $\nabla \times \mathbf{E} = - \mu \frac{\partial \mathbf{H}} {\partial t}$ | 磁場強度 H 的變化會產生感應電場 E
安培定律 		| $\nabla \times \mathbf{H} = \mathbf{J} + \epsilon \frac{\partial \mathbf{E}} {\partial t}$ | 電流 J 與電場強度 E 的變化 $\frac{\partial \mathbf{E}} {\partial t}$  會產生磁場 H
