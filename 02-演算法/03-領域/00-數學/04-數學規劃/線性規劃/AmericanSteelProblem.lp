\* American_Steel_Problem *\
Minimize
Total_Cost_of_Transport: 0.12 Route_('Chicago',_'Gary')
 + 0.6 Route_('Chicago',_'Tempe') + 0.35 Route_('Cincinatti',_'Albany')
 + 0.55 Route_('Cincinatti',_'Houston')
 + 0.375 Route_('Kansas_City',_'Houston')
 + 0.65 Route_('Kansas_City',_'Tempe') + 0.4 Route_('Pittsburgh',_'Chicago')
 + 0.35 Route_('Pittsburgh',_'Cincinatti') + 0.45 Route_('Pittsburgh',_'Gary')
 + 0.45 Route_('Pittsburgh',_'Kansas_City')
 + 0.5 Route_('Youngstown',_'Albany') + 0.375 Route_('Youngstown',_'Chicago')
 + 0.35 Route_('Youngstown',_'Cincinatti')
 + 0.45 Route_('Youngstown',_'Kansas_City')
Subject To
Steel_Flow_Conservation_in_Node_Albany: Route_('Cincinatti',_'Albany')
 + Route_('Youngstown',_'Albany') >= 3000
Steel_Flow_Conservation_in_Node_Chicago: - Route_('Chicago',_'Gary')
 - Route_('Chicago',_'Tempe') + Route_('Pittsburgh',_'Chicago')
 + Route_('Youngstown',_'Chicago') >= 0
Steel_Flow_Conservation_in_Node_Cincinatti: - Route_('Cincinatti',_'Albany')
 - Route_('Cincinatti',_'Houston') + Route_('Pittsburgh',_'Cincinatti')
 + Route_('Youngstown',_'Cincinatti') >= 0
Steel_Flow_Conservation_in_Node_Gary: Route_('Chicago',_'Gary')
 + Route_('Pittsburgh',_'Gary') >= 6000
Steel_Flow_Conservation_in_Node_Houston: Route_('Cincinatti',_'Houston')
 + Route_('Kansas_City',_'Houston') >= 7000
Steel_Flow_Conservation_in_Node_Kansas_City:
 - Route_('Kansas_City',_'Houston') - Route_('Kansas_City',_'Tempe')
 + Route_('Pittsburgh',_'Kansas_City') + Route_('Youngstown',_'Kansas_City')
 >= 0
Steel_Flow_Conservation_in_Node_Pittsburgh: - Route_('Pittsburgh',_'Chicago')
 - Route_('Pittsburgh',_'Cincinatti') - Route_('Pittsburgh',_'Gary')
 - Route_('Pittsburgh',_'Kansas_City') >= -15000
Steel_Flow_Conservation_in_Node_Tempe: Route_('Chicago',_'Tempe')
 + Route_('Kansas_City',_'Tempe') >= 4000
Steel_Flow_Conservation_in_Node_Youngstown: - Route_('Youngstown',_'Albany')
 - Route_('Youngstown',_'Chicago') - Route_('Youngstown',_'Cincinatti')
 - Route_('Youngstown',_'Kansas_City') >= -10000
Bounds
0 <= Route_('Chicago',_'Gary') <= 4000
0 <= Route_('Chicago',_'Tempe') <= 2000
1000 <= Route_('Cincinatti',_'Albany') <= 5000
0 <= Route_('Cincinatti',_'Houston') <= 6000
0 <= Route_('Kansas_City',_'Houston') <= 4000
0 <= Route_('Kansas_City',_'Tempe') <= 4000
0 <= Route_('Pittsburgh',_'Chicago') <= 4000
0 <= Route_('Pittsburgh',_'Cincinatti') <= 2000
0 <= Route_('Pittsburgh',_'Gary') <= 2000
2000 <= Route_('Pittsburgh',_'Kansas_City') <= 3000
0 <= Route_('Youngstown',_'Albany') <= 1000
0 <= Route_('Youngstown',_'Chicago') <= 5000
0 <= Route_('Youngstown',_'Cincinatti') <= 3000
1000 <= Route_('Youngstown',_'Kansas_City') <= 5000
Generals
Route_('Chicago',_'Gary')
Route_('Chicago',_'Tempe')
Route_('Cincinatti',_'Albany')
Route_('Cincinatti',_'Houston')
Route_('Kansas_City',_'Houston')
Route_('Kansas_City',_'Tempe')
Route_('Pittsburgh',_'Chicago')
Route_('Pittsburgh',_'Cincinatti')
Route_('Pittsburgh',_'Gary')
Route_('Pittsburgh',_'Kansas_City')
Route_('Youngstown',_'Albany')
Route_('Youngstown',_'Chicago')
Route_('Youngstown',_'Cincinatti')
Route_('Youngstown',_'Kansas_City')
End
