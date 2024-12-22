

來源 -- [1D Electromagnetic FDTD in Python (ipynb)](https://github.com/natsunoyuki/blog_posts/blob/main/physics/1D%20Electromagnetic%20FDTD%20in%20Python.ipynb)


ChatGPT 中文翻譯

https://chatgpt.com/c/6752b708-1d14-8012-8bbb-d4e206452f7e

### 1D 電磁 FDTD 在 Python 中的實作

有限差分時域法（FDTD）是一種數值方法，用來解決包含空間和時間導數的偏微分方程，如麥克斯韋方程組或地球內部傳播的地震波。這些方程組對於複雜的系統（例如隨機激光器或火山系統）無法進行解析解，因此科學家必須利用數值方法來更好地理解這些系統在不同條件下的行為。本文將介紹如何在 Python 中實現一維電磁 FDTD 演算法。

### 麥克斯韋方程組

由於本文重點在電磁學，我們從高斯單位制下的麥克斯韋方程組開始，並將單位縮放使得光速 \(c = 1\)，並假設系統中沒有電流 \(J\)：

\[
\nabla \cdot \mathbf{E} = 4\pi \rho,
\]
\[
\nabla \cdot \mathbf{B} = 0,
\]
\[
\nabla \times \mathbf{E} = - \frac{\partial \mathbf{B}}{\partial t},
\]
\[
\nabla \times \mathbf{B} = \frac{\partial \mathbf{E}}{\partial t},
\]

其中 \( \frac{\partial \mathbf{B}}{\partial t} \) 和 \( \frac{\partial \mathbf{E}}{\partial t} \) 分別是磁場和電場的時間導數。

### 從麥克斯韋方程組到 FDTD

為了實現 FDTD 演算法，我們只需要使用第三和第四條方程。第三條方程可以寫成：

\[
(\nabla \times \mathbf{E})_i = \epsilon_{ijk} \nabla_j E_k = - \frac{\partial \mathbf{B}_i}{\partial t},
\]
其中 \(i, j, k \in \{x, y, z\}\)， \( \epsilon_{ijk} \) 是 Levi-Civita 符號，\( \nabla_j \) 是第 \(j\) 空間導數，\( E_k \) 是第 \(k\) 分量的電場。對於一維系統，其中只有 \( \nabla_x \) 存在，這簡化為兩個方程：

\[
-\nabla_x E_z = - \frac{\partial B_y}{\partial t},
\]
\[
\nabla_x E_y = - \frac{\partial B_z}{\partial t}.
\]

第四條方程可以寫成：

\[
(\nabla \times \mathbf{B})_i = \epsilon_{ijk} \nabla_j B_k = \frac{\partial \mathbf{E}_i}{\partial t},
\]

對於一維系統，這簡化為：

\[
-\nabla_x B_z = \frac{\partial E_y}{\partial t},
\]
\[
\nabla_x B_y = \frac{\partial E_z}{\partial t}.
\]

這四條方程可以重組成兩組耦合的偏微分方程。在第一組中，我們只有 \( E_z \) 和 \( B_y \)：

\[
-\nabla_x E_z = - \frac{\partial B_y}{\partial t},
\]
\[
\nabla_x B_y = \frac{\partial E_z}{\partial t},
\]

在第二組中，我們只有 \( B_z \) 和 \( E_y \)：

\[
\nabla_x E_y = - \frac{\partial B_z}{\partial t},
\]
\[
-\nabla_x B_z = \frac{\partial E_y}{\partial t}.
\]

由於這兩組耦合方程在空間和時間對稱下基本相同，因此我們只需選擇並數值實現其中一組，即可實現 1D FDTD 演算法。

### FDTD 網格

現在我們知道要解決的方程，接下來需要對耦合的電場和磁場進行離散化。理論上，這些耦合方程必須同時解決，但在數值解中這顯然是不可能的。因此，在 FDTD 方法中，我們使用交錯的空間和時間網格來離散化方程，這樣可以數值上近似同時解決這兩個耦合方程。

在交錯的 FDTD 網格中，電場（藍色網格點）在整數空間指標 \( ..., i - 1, i, i + 1, ... \) 和整數時間指標 \( ..., n - 1, n, n + 1, ... \) 上進行評估，而磁場（紅色網格點）在半整數空間指標 \( ..., i - 1/2, i + 1/2, ... \) 和半整數時間指標 \( ..., n - 1/2, n + 1/2, ... \) 上進行評估。

使用交錯的 FDTD 網格，第一組方程：

\[
-\nabla_x E_z = - \frac{\partial B_y}{\partial t},
\]
\[
\nabla_x B_y = \frac{\partial E_z}{\partial t},
\]

可以離散化為：

\[
-(E_z[i, n] - E_z[i - 1, n])/Δx = -(B_y[i - 1/2, n + 1/2] - B_y[i - 1/2, n - 1/2])/Δt,
\]

這可以重組為：

\[
B_y[i - 1/2, n + 1/2] = B_y[i - 1/2, n - 1/2] - Δt (E_z[i, n] - E_z[i, n - 1])/Δx.
\]

同樣，方程 \( \nabla_x B_y = \frac{\partial E_z}{\partial t} \) 可以離散化為：

\[
(E_z[i, n] - E_z[i - 1, n])/Δx = (B_y[i + 1/2, n - 1/2] - B_y[i - 1/2, n - 1/2])/Δt,
\]

並重組為：

\[
E_z[i, n] = E_z[i, n - 1] - Δt (B_y[i + 1/2, n - 1/2] - B_y[i - 1/2, n - 1/2])/Δx.
\]

同樣，第二組方程：

\[
\nabla_x E_y = - \frac{\partial B_z}{\partial t},
\]
\[
-\nabla_x B_z = \frac{\partial E_y}{\partial t},
\]

可以離散化並重組為：

\[
B_z[i - 1/2, n + 1/2] = B_z[i - 1/2, n - 1/2] - Δt (E_y[i, n] - E_y[i - 1, n])/Δx,
\]
\[
E_y[i, n] = E_y[i, n - 1] - Δt (B_z[i + 1/2, n - 1/2] - B_z[i - 1/2, n - 1/2])/Δx.
\]

### FDTD 演算法的 Python 實現

整個 1D FDTD 模擬器被編碼為一個 `fdtd1d` 類。當初始化時，會生成 FDTD 網格和模擬參數。調用 `run` 會執行對電場和磁場的更新。

這裡的電場更新排除了離散網格中的首尾元素，這樣可以使兩端的電場始終為零，這對應於完美的電導體。一個電磁波撞擊完美的電導體時，將完全反射回來，將任何生成的電磁波完全困在 FDTD 網格內。

在數值模擬穩定性方面，更新步長必須符合 Courant 條件，即時間步長應滿足：

\[
dt = dx / (1.0 * c).
\]

當執行模擬時，我們還需要在網格中引入源，以啟動數值求解。我們通過將生成的數值加到所需源網格點的電場中來引入源。

### 吸收邊界條件

如果我們不希望將生成的電磁波困在網格內，而是允許它在網格端點處消散，我們需要修改網格並引入吸收邊界條件。吸收邊界條件將吸收任何撞擊網格端點的電磁波，允許波從網格中消散。在這裡，我們直接使用 Mur 吸收邊界條件，感興趣的讀者可以
