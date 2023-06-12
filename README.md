# Terraform

Terrafrom [QGIS](https://www.qgis.org/en/site/) plugin contains implementation of 11 popular topographic correction algorithms. 
Besides it the plugin provides an ability to evaluate the results of the correction algorithms by several methods: 
correlation between luminance and image digital numbers, rose-diagrams analysis and multi-criteria statistical analysis. 
Also, the plugin is capable of ranking topographic correction algorithms according to their applicability to a particular 
image using multi-criteria score based on the statistical metrics. Terraform plugin has processing provider, 
so you can easily combine the algorithm described above with other processing algorithms.

### Implemented topographic correction methods

1. [Cosine-T](https://www.tandfonline.com/doi/abs/10.1080/07038992.1982.10855028) 
2. [Cosine-C](https://www.asprs.org/wp-content/uploads/pers/1989journal/sep/1989_sep_1303-1309.pdf) 
3. [C-correction](https://www.tandfonline.com/doi/abs/10.1080/07038992.1982.10855028) 
4. [SCS](http://dx.doi.org/10.1016/S0034-4257(97)00177-6) 
5. [SCS + C](http://dx.doi.org/10.1109/TGRS.2005.852480) 
6. [Minnaert](https://www.asprs.org/wp-content/uploads/pers/1980journal/sep/1980_sep_1183-1189.pdf) 
7. [Minnaert-SCS](https://ui.adsabs.harvard.edu/abs/2002PhDT........92R/abstract) 
8. [PBM](https://www.researchgate.net/publication/235244169_Pixel-based_Minnaert_Correction_Method_for_Reducing_Topographic_Effects_on_a_Landsat_7_ETM_Image) 
9. [VECA](https://ieeexplore.ieee.org/abstract/document/4423917/) 
10. [Teillet regression](https://www.tandfonline.com/doi/abs/10.1080/07038992.1982.10855028)
11. [Pixel-based C correction](https://www.tandfonline.com/doi/full/10.1080/01431160701881889)
