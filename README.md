# DataCastle_HI-GUIDES
精品旅行服务成单预测比赛是DC举办的“第二届智慧中国杯”的第一个比赛，主要是通过用户的历史行为数据对待预测用户是否购买精品旅游服务进行预测。具体的比赛背景和数据介绍可以参考比赛说明。

最近半年也参加了几个比赛，但是成绩也并不是很突出，所以也一直没有做过总结。趁着过年这几天比较空，将“皇包车”比赛的思路做一个小结，方便自己以后查看。比赛最后B榜的成绩为0.95630，排名79/1073 

比赛的大神很多，自己写的可能存在很多不足之处，希望能和大家多多交流。

-------------- 分割线 -------------------

data文件夹下train和test文件夹分别放置下载的训练和测试数据。data文件夹下的其他文件分别为特征工程提取的特征（train and test），初期提交的文件以及特征重要性文档。这里提交的结果只是最开始的结果（对应model中的baseline），并不是最后调参的结果。

feature_extraction 和 model文件夹下分别放置特征工程和模型的code。
