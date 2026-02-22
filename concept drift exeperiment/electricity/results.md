

[A] 滚动 P(UP) 的时序相关系数（r 越接近 1 说明漂移趋势越吻合）
  w= 50  r=+0.0639  p=4.73e-03
  w=100  r=+0.1631  p=8.30e-13
  w=200  r=+0.1210  p=2.60e-07

[B] 各特征滚动均值（w=100）的时序相关系数
  date          r=+0.4864
  day           r=-0.2105
  period        r=-0.0008
  nswprice      r=+0.1431
  nswdemand     r=+0.1310
c:\reposity_clone\be_great\evaluate_drift.py:201: ConstantInputWarning: An input array is constant; the correlation coefficient is not defined.
  rho, _ = stats.pearsonr(r_m.values[:n], s_m.values[:n])
  vicprice      r=+nan
  vicdemand     r=+nan
  transfer      r=+nan

[C] 分段分类准确率
  Real  均值: 0.8106  各段: ['0.645', '0.755', '0.905', '0.875', '0.805', '0.940', '0.955', '0.745', '0.670']
  Synth 均值: 0.7883  各段: ['0.820', '0.810', '0.745', '0.790', '0.775', '0.830', '0.795', '0.765', '0.765']
  rho, _ = stats.pearsonr(r_m.values[:n], s_m.values[:n])
  vicprice      r=+nan
  vicdemand     r=+nan
  transfer      r=+nan

[C] 分段分类准确率
  Real  均值: 0.8106  各段: ['0.645', '0.755', '0.905', '0.875', '0.805', '0.940', '0.955', '0.745', '0.670']
  rho, _ = stats.pearsonr(r_m.values[:n], s_m.values[:n])
  vicprice      r=+nan
  vicdemand     r=+nan
  transfer      r=+nan

[C] 分段分类准确率
  rho, _ = stats.pearsonr(r_m.values[:n], s_m.values[:n])
  vicprice      r=+nan
  vicdemand     r=+nan
  transfer      r=+nan
  rho, _ = stats.pearsonr(r_m.values[:n], s_m.values[:n])
  vicprice      r=+nan
  vicdemand     r=+nan
  rho, _ = stats.pearsonr(r_m.values[:n], s_m.values[:n])
  vicprice      r=+nan
  vicdemand     r=+nan
  rho, _ = stats.pearsonr(r_m.values[:n], s_m.values[:n])
  vicprice      r=+nan
  rho, _ = stats.pearsonr(r_m.values[:n], s_m.values[:n])
  rho, _ = stats.pearsonr(r_m.values[:n], s_m.values[:n])
  rho, _ = stats.pearsonr(r_m.values[:n], s_m.values[:n])
  rho, _ = stats.pearsonr(r_m.values[:n], s_m.values[:n])
  rho, _ = stats.pearsonr(r_m.values[:n], s_m.values[:n])
  vicprice      r=+nan
  rho, _ = stats.pearsonr(r_m.values[:n], s_m.values[:n])
  vicprice      r=+nan
  rho, _ = stats.pearsonr(r_m.values[:n], s_m.values[:n])
  rho, _ = stats.pearsonr(r_m.values[:n], s_m.values[:n])
  rho, _ = stats.pearsonr(r_m.values[:n], s_m.values[:n])
  vicprice      r=+nan
  vicdemand     r=+nan
  transfer      r=+nan

[C] 分段分类准确率
  Real  均值: 0.8106  各段: ['0.645', '0.755', '0.905', '0.875', '0.805', '0.940', '0.955', '0.745', '0.670']
  Synth 均值: 0.7883  各段: ['0.820', '0.810', '0.745', '0.790', '0.775', '0.830', '0.795', '0.765', '0.765']
  MAE (Real vs Synth): 0.0989  （越小说明合成数据的漂移程度越接近真实）

[D] 全局 KS 检验（stat 越小说明合成数据整体分布越接近真实）
  date          KS=0.9595  p=0.00e+00
  day           KS=0.0680  p=1.92e-04
  period        KS=0.0400  p=8.15e-02
  nswprice      KS=0.3580  p=3.16e-114
  nswdemand     KS=0.1325  p=1.03e-15
  vicprice      KS=0.9140  p=0.00e+00
  vicdemand     KS=0.7315  p=0.00e+00
  transfer      KS=0.9290  p=0.00e+00