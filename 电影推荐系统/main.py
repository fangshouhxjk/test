from math import sqrt

critics = {'老炮儿': {'A': 3.5, 'B': 2.5, 'C': 3.0, 'D': 2.5, 'E': 3.5, 'F': 3.0, 'G': 4.5},
           '唐人街探案': {'A': 1.0, 'B': 3.5, 'C': 3.5, 'D': 3.5, 'E': 2.0, 'F': 4.0, 'G': 1.5},
           '星球大战': {'B': 3.0, 'C': 1.5, 'E': 4.5, 'F': 2.0, 'G': 3.0},
           '寻龙诀': {'B': 3.5, 'C': 5.0, 'D': 3.5, 'F': 3.0, 'G': 5.0},
           '神探夏洛克': {'B': 2.5, 'C': 3.0, 'D': 4.0, 'E': 3.5, 'F': 3.0, 'G': 3.5},
           '小门神': {'B': 3.0, 'C': 3.5, 'E': 2.0, 'F': 2.0}
           }
print(critics['老炮儿']['A'])

'''皮尔逊相关系数计算'''
def sim_pearson(prefs, p1, p2):
    si = {}
    for item in prefs[p1]:
        if item in prefs[p2]: si[item] = 1
    if len(si) == 0: return 0
    n = len(si)
    sum1 = sum([prefs[p1][it] for it in si])
    sum2 = sum([prefs[p2][it] for it in si])
    sum1Sq = sum([pow(prefs[p1][it], 2) for it in si])
    sum2Sq = sum([pow(prefs[p2][it], 2) for it in si])
    pSum = sum([prefs[p1][it] * prefs[p2][it] for it in si])
    num = pSum - (sum1 * sum2 / n)
    den = sqrt((sum1Sq - pow(sum1, 2) / n) * (sum2Sq - pow(sum2, 2) / n))
    if den == 0:
        return 0
    r = num / den
    return r
print(sim_pearson(critics,'唐人街探案','神探夏洛克'))#得出唐人街探案 和神探夏洛克的皮尔逊系数对比值
#编写函数依据Pearson相关系数大小以及协同过滤算法（物品）实现电影的推荐。
'''进行加权，归一，排序'''
def getRecommendations(prefs, person, similarity=sim_pearson):
    totals = {}
    simSums = {}
    for other in prefs:
        if other == person: continue
        sim = similarity(prefs, person, other)
        if sim <= 0: continue
        for item in prefs[other]:
            if item not in prefs[person] or prefs[person][item] == 0:
                totals.setdefault(item, 0)
                totals[item] += prefs[other][item] * sim
                simSums.setdefault(item, 0)
                simSums[item] += sim
    rankings = [(total / simSums[item], item) for item, total in totals.items()]
    rankings.sort()
    rankings.reverse()
    return rankings
print(getRecommendations(critics,'星球大战'))

'''进行计算然后推荐'''
def transformPrefs(prefs):
    result = {}
    for person in prefs:
        for item in prefs[person]:
            result.setdefault(item, {})
            result[item][person] = prefs[person][item]
    return result
movies = transformPrefs(critics)
adviseMovie = getRecommendations(movies, 'A')
print(adviseMovie)
Threshold =4
for item in adviseMovie:
    if item[0] >Threshold:
        print('给A推荐的电影是：'+ item[1])