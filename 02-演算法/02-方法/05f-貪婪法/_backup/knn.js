def knn():
  KNN = :

  KNN.loadQA = def (QA) :
    KNN.QA = QA
  

  KNN.kNearestNeighbors = def (item, distance, k = 1) :
    var QA = KNN.QA
    var d = []
    for (var i = 0  i < QA.length  i++) :
      d.push(:dist: distance(item, QA[i].Q), qa: QA[i])
    
    d.sort((o1, o2) => o1.dist > o2.dist)
    return d.slice(0, k)
  

  return KNN

