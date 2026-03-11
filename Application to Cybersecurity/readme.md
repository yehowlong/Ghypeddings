## 1. Overview
This section focuses on comparing graph hyperbolic embedding on detecting threats and attacks on communication traces represented as graphs. 

## 2. Citation
The results of this comparaison have been published in:
Mohamed Yacine Touahria Miliani, Souhail Abdelmouaiz Sadat, Mohammed Haddad, Hamida Seba, and Karima Amrouche. 2024. Comparing Hyperbolic Graph Embedding models on Anomaly Detection for Cybersecurity. In Proceedings of the 19th International Conference on Availability, Reliability and Security (ARES '24). Association for Computing Machinery, New York, NY, USA, Article 118, 1–11. https://doi.org/10.1145/3664476.3670445

## 3. Datasets

The following intrusion detection datasets were used to test and evaluate the models implemented in the library. Our code includes all the pre-processing steps required to convert these datasets from tabular format into graphs. Due to usage restrictions, this library provides only a single graph of each dataset, with 5,000 nodes, already pre-processed and normalized.

| Name                   | Ref   |
|------------------------|-------|
| CIC-DDoS2019           | [7]   |
| AWID3                  | [8]   |
| CIC-Darknet2020        | [9]   |
| NF-UNSW-NB15-V2        | [10]  |
| NF-BoT-IoT-V2          | [11]  |
| NF-CSE-CIC-IDS2018-V2  | [12]  |

---

## References  

[7] CIC-DDoS2019 dataset. Available at: [https://www.unb.ca/cic/datasets/ddos-2019.html](https://www.unb.ca/cic/datasets/ddos-2019.html)  

[8] AWID3 dataset. Available at: [https://icsdweb.aegean.gr/awid/](https://icsdweb.aegean.gr/awid/)  

[9] CIC-Darknet2020 dataset. Available at: [https://www.unb.ca/cic/datasets/darknet2020.html](https://www.unb.ca/cic/datasets/darknet2020.html)  

[10] NF-UNSW-NB15-V2 dataset. Available at: [https://staff.itee.uq.edu.au/marius/NIDS_datasets/#RA6](https://staff.itee.uq.edu.au/marius/NIDS_datasets/#RA6)  

[11] NF-BoT-IoT-V2 dataset. Available at: [https://staff.itee.uq.edu.au/marius/NIDS_datasets/#RA8](https://staff.itee.uq.edu.au/marius/NIDS_datasets/#RA8)  

[12] NF-CSE-CIC-IDS2018-V2 dataset. Available at: [https://staff.itee.uq.edu.au/marius/NIDS_datasets/#RA9](https://staff.itee.uq.edu.au/marius/NIDS_datasets/#RA9)  
