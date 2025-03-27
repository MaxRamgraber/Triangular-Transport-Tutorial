import scipy.stats
import numpy as np
import matplotlib.pyplot as plt
import copy
import itertools
import math
import pickle
import matplotlib
from matplotlib import colors
import os
from matplotlib import gridspec
from transport_map import *

# Colors of the colormap
turbocolors = [[0.6904761904761906, 0.010977778700493195, 0.012597277894375276], [0.6906519235270884, 0.01640135787405724, 0.0837140289089694], [0.6917028231470678, 0.0229766531631339, 0.1508422093503125], [0.6935600550923448, 0.03059217460463138, 0.2140806913791454], [0.6961547851191345, 0.03913937310725368, 0.27354760270667333], [0.6994181789836523, 0.048510190139500135, 0.32937485403121564], [0.7032814024421139, 0.05859488216300645, 0.3817042734043511], [0.7076756212507346, 0.06928011981122602, 0.4306849504666904], [0.7125320011657298, 0.08044736181345138, 0.4764714640882298], [0.717781707943315, 0.09197150366417624, 0.5192227314969523], [0.7233559073397057, 0.10371980103779682, 0.5591012754819024], [0.7291857651111174, 0.11555106794865597, 0.5962727587134745], [0.7352024470137656, 0.12731514965642451, 0.6309056806339336], [0.7413371188038657, 0.13885267031682524, 0.6631711727355227], [0.7475209462376329, 0.1499950553776956, 0.6932428623615332], [0.7536850950712828, 0.16056482872039282, 0.7212968034377881], [0.7597607310610313, 0.17037618454653694, 0.747511494767853], [0.75929001722113, 0.17923583401009582, 0.7656790199630931], [0.7475941038279283, 0.18694820855183453, 0.7713723012300058], [0.7366824556818949, 0.19346062714094175, 0.7768150322166271], [0.72643042874709, 0.19890811745519935, 0.7820344349913866], [0.7167412048139664, 0.20344443760235664, 0.7870610686424058], [0.7075377277455205, 0.20722973652712667, 0.7919254922578067], [0.6987607996238635, 0.2104298456332004, 0.7966582649257115], [0.6903673228438931, 0.21321570323575734, 0.8012899457342416], [0.682328749433452, 0.21576291184447535, 0.8058510937715191], [0.6746297909082095, 0.21825142827703536, 0.8103722681256655], [0.6672648697027385, 0.22085425646468607, 0.8148817689184499], [0.6602014830869536, 0.22359379565318152, 0.8193767173456075], [0.6533787124237928, 0.2263796531215008, 0.823831507046331], [0.6467289333281976, 0.22911680044368107, 0.8282200161103943], [0.6401785662494847, 0.23170815719718751, 0.8325161226275704], [0.6336482569987627, 0.23405470433309905, 0.8366937046876322], [0.6270530344594542, 0.2360556532010278, 0.8407266403803533], [0.6203024476840878, 0.23760867022877552, 0.8445888077955067], [0.6133006844761989, 0.23861015725671983, 0.8482540850228658], [0.6059624573538679, 0.23898238778533165, 0.8516979593044193], [0.5983090374215976, 0.23882956412204973, 0.854906810971959], [0.5904174130986337, 0.23833421958277098, 0.8578715372512724], [0.5823783687413812, 0.2376823588176445, 0.8605830499184942], [0.5742958438497263, 0.23706268520314622, 0.8630322607497598], [0.566286260437254, 0.23666602945460796, 0.8652100815212042], [0.5584776771672347, 0.23668473807742815, 0.8671074240089628], [0.5510087685640095, 0.23731202165697124, 0.8687151999891709], [0.5440276258636073, 0.23874126128013823, 0.8700243216224771], [0.5376105577619357, 0.2411191644739669, 0.8710360813847914], [0.5315489266889114, 0.24442474436899914, 0.871789217558878], [0.5255684953009153, 0.24859807924582497, 0.8723309092587582], [0.5193987629250915, 0.2535784508499333, 0.8727083355984544], [0.5127774500657107, 0.25930498111025313, 0.8729686756919878], [0.5054539190109256, 0.2657171896860476, 0.8731591086533795], [0.4971915396084222, 0.2727554723421566, 0.8733268135966521], [0.4877690201269585, 0.2803615001525919, 0.8735189696358265], [0.47698218055553143, 0.28847794895571527, 0.873782586986574], [0.4648276728474042, 0.29697110893585776, 0.8741423773429315], [0.45165101189669743, 0.305552156792384, 0.8745787059949128], [0.43777753809619335, 0.31391252849082757, 0.8750666949974303], [0.4234601983798163, 0.32174200618204823, 0.8755814664053952], [0.4088786643397557, 0.3287286961969925, 0.87609814227372], [0.39414368224229085, 0.33455917871529023, 0.8765918446573163], [0.3793067492830514, 0.3389188291076908, 0.877037695611096], [0.3643751994164286, 0.34149231095233595, 0.8774108171899714], [0.34933070330797267, 0.34197454203820826, 0.877688092213006], [0.3403673509225958, 0.34663277097930234, 0.8778987375452829], [0.3370236501326216, 0.3554484329299287, 0.8781318704949722], [0.33231461663862033, 0.3618511397272365, 0.8784798910693628], [0.3266091000188368, 0.36659036039709064, 0.8790351992757446], [0.32027293973512316, 0.3703929847930015, 0.8798901951214063], [0.31366899325232445, 0.37397422952969406, 0.8811372786136377], [0.3071578750013543, 0.3780456944328393, 0.8828688497597277], [0.301099406185961, 0.38332054751916245, 0.8851773085669661], [0.29583267832568955, 0.3904653314042777, 0.8881489717217561], [0.2914323797009556, 0.39953599160062525, 0.8917964039595472], [0.2877980734089111, 0.41017448566248715, 0.8960831069995929], [0.28482805439516534, 0.4220062674230746, 0.9009716956503901], [0.28242172761403994, 0.43464761751547915, 0.906424784720434], [0.2804788409846796, 0.4477051726897616, 0.9124049890182206], [0.2788987912262992, 0.4607749922480771, 0.918874923352247], [0.277580002572562, 0.47344114547803795, 0.9257972025310082], [0.2764193783650953, 0.485273787396179, 0.9331344413630012], [0.27531382596760734, 0.49590948909381166, 0.9408321707166999], [0.27417124253611536, 0.5054794081653604, 0.9487320904666496], [0.2729055061729636, 0.5143187877995266, 0.9566368878753057], [0.27143253536604345, 0.5227855656932154, 0.9643491802293045], [0.2696704730654312, 0.531261581824037, 0.9716715848152837], [0.2675399881902376, 0.5401524972415588, 0.9784067189198801], [0.2649646868587828, 0.5498860194147946, 0.9843571998297309], [0.26187163334209196, 0.5609083493724006, 0.9893256448314729], [0.25819198579084385, 0.5736786941767217, 0.9931146823187612], [0.25388033135389326, 0.5884766343804858, 0.9955682574046939], [0.2489559746899319, 0.6049657034015569, 0.9966648267466579], [0.24345275591304505, 0.62265633216535, 0.9964101744320204], [0.23740633990537824, 0.6410301303413559, 0.9948100845481483], [0.23085404063102094, 0.6595442628994237, 0.9918703411824094], [0.22383464684219176, 0.6776362279674177, 0.9875967284221705], [0.21638824917772453, 0.6947290781287391, 0.9819950303547987], [0.20855606865385165, 0.7102371206640654, 0.9750710310676618], [0.20038059410110376, 0.7235758485222823, 0.9668315591455693], [0.19193351310774223, 0.734519649068058, 0.957382599029186], [0.1833340368744243, 0.7435006401649751, 0.9470095683956252], [0.17469783713168363, 0.7510390693547768, 0.9360167776775004], [0.16613077624888986, 0.7576720124478675, 0.9247085373074251], [0.15772857678376886, 0.763950628076615, 0.9133891577180121], [0.14957672814523254, 0.7704390486819581, 0.9023629493418752], [0.14175063036952132, 0.7777150132050495, 0.8919342226116274], [0.1343159750096613, 0.7863723484029234, 0.8824072879598819], [0.12732970449680675, 0.797012529379906, 0.8740811995081446], [0.12084976147141473, 0.8099384376963948, 0.8671237357742304], [0.11493975728845535, 0.8251268658709435, 0.8615653598949594], [0.10966332888119507, 0.8425408262745097, 0.857430069974358], [0.1050853463044867, 0.8547418641164511, 0.8473403551838433], [0.10127356192382456, 0.8535247404252649, 0.8231538838024755], [0.09830031072117495, 0.8538026970048249, 0.7998509537157463], [0.09624426171757831, 0.8555997319591567, 0.7775291500837249], [0.09519222051252364, 0.8589398433922859, 0.7562951678279484], [0.09523466033004026, 0.8638301588380453, 0.736254113762426], [0.09640562817698234, 0.8700971873969563, 0.7173867121266534], [0.09870775682645283, 0.8774578722514202, 0.6996232876010631], [0.10214964316230982, 0.8856276192531288, 0.6829345377608849], [0.10674326864505639, 0.8943218342537755, 0.6673309599875173], [0.1125007767162798, 0.9032559231050521, 0.6528591156909103], [0.1194311818691091, 0.9121452916586508, 0.6395960113839659], [0.12753701038469437, 0.9207053457662641, 0.6276416598987021], [0.13681087273470835, 0.9286514912795846, 0.6171099273803513], [0.1472273301533093, 0.9357451866752111, 0.608020030012468], [0.1587227785675077, 0.9419996156076766, 0.5998477287978358], [0.17121964482967164, 0.9475135374443872, 0.5918796184356399], [0.18463807709930177, 0.952385782298133, 0.5834130936511204], [0.19889759168666765, 0.9567151802817035, 0.5737737807193738], [0.21391852693235652, 0.9606005615078892, 0.562329072946783], [0.22962331181780002, 0.9641407560894797, 0.5484979436976313], [0.24593754930678074, 0.9674345941392647, 0.5317571316448257], [0.26279082355117567, 0.9705808834297661, 0.511644132353509], [0.2800068036776241, 0.9736514082827111, 0.48808838425690876], [0.2970834223782295, 0.9766384622071741, 0.4619539474027109], [0.31345165544585374, 0.9795197960248092, 0.4340817805575162], [0.3285349032795682, 0.982273160557271, 0.40507369070122135], [0.3417498926656193, 0.9848763066262134, 0.3752866956776463], [0.36016989144758, 0.9873069850532904, 0.35250783102398103], [0.40675615393016756, 0.9895429466601561, 0.36021581312049306], [0.44700613398058514, 0.9915619422684648, 0.3642784802445829], [0.4800453603427492, 0.9933416833596941, 0.3641065514694242], [0.5059172225697819, 0.9948570776317582, 0.3595849281383], [0.5260321807100684, 0.996078410770516, 0.35138497368822147], [0.5417361711468346, 0.9969755367452103, 0.340261882906466], [0.5542251246559543, 0.9975183095250835, 0.3269807394030567], [0.564594836019787, 0.9976765830793782, 0.3123146193759361], [0.5738773907037504, 0.9974202113773375, 0.29704222917768014], [0.5830638840476217, 0.9967190483882035, 0.28194507668375485], [0.593113257901611, 0.9955429480812196, 0.2678041764623123], [0.6049173329252986, 0.9938631942887316, 0.2553741911685384], [0.618688941513129, 0.9916814407117477, 0.24493286081566687], [0.6340191489156332, 0.9890283751100516, 0.23630248008060123], [0.6504815845022238, 0.9859358477750884, 0.22928282597582322], [0.667664607259613, 0.9824357089983022, 0.2236706549395957], [0.6851775225926456, 0.9785598090711378, 0.21926108516565482], [0.7026554225327669, 0.9743399982850405, 0.21584884718485342], [0.7197626647068687, 0.9698081269314549, 0.21322940269875598], [0.7361950207876023, 0.9649960453018253, 0.21119993166518547], [0.7517045578243673, 0.9599385848130461, 0.20957252075787536], [0.7662987145771546, 0.9546988900091035, 0.20827881408159643], [0.7801235262514455, 0.9493557514336002, 0.20731561723168843], [0.793333881434332, 0.9439881244434671, 0.2066805054707251], [0.8060910507633685, 0.9386749643956355, 0.20637143421171145], [0.818561771106568, 0.9334952266470367, 0.20638701190028513], [0.8309175422374003, 0.9285278665546012, 0.20672674995777193], [0.8433341375472628, 0.9238518394752603, 0.2073912897850912], [0.8559913280771064, 0.9195461007659449, 0.20838260682751408], [0.8690561732865327, 0.9156936485066011, 0.20969336805791894], [0.8826222717866333, 0.9123973799376971, 0.21126511049859706], [0.8967754117130945, 0.9097663045485556, 0.21302599542691433], [0.9079094334716317, 0.9041987442580517, 0.21490680311671295], [0.9069357778393816, 0.886592094707027, 0.21684014643856891], [0.9069543487842606, 0.870013975207323, 0.21875944038476458], [0.9080741574387239, 0.8545231793863904, 0.22059763194219112], [0.910404214935227, 0.8401608509928674, 0.22228569031318043], [0.9140533609890943, 0.8269505724832292, 0.22375088679927074], [0.9190348675863581, 0.8148283391986565, 0.22493134638188253], [0.9251076558303858, 0.8035443408635032, 0.22580368730895622], [0.9319887969388863, 0.7928444208549985, 0.2263494470448247], [0.9393953621295686, 0.7825040250835419, 0.2265502781157535], [0.9470444226201423, 0.7723286488430519, 0.2263898716540945], [0.9546530496283158, 0.762153889073118, 0.22585549404541186], [0.9619383143717981, 0.7518450142735859, 0.2249391366785828], [0.9686172880682985, 0.7412959626121468, 0.22363827879886775], [0.9744096708096767, 0.730428526919429, 0.22195619461848926], [0.9791805893769406, 0.7192323214790531, 0.21989751414051684], [0.9830137772099842, 0.707766047135114, 0.21746224880159565], [0.9860109991965004, 0.6960819371591295, 0.21464892131240673], [0.9882740202241823, 0.6842207640296426, 0.21145507809451805], [0.9899046051807234, 0.6722129383813515, 0.2078773477615485], [0.9910045189538177, 0.660079441906903, 0.203911434685365], [0.9916755264311576, 0.6478325847934217, 0.19955204764730958], [0.9920193925004372, 0.6354765794026785, 0.19479276357445865], [0.9921346225982788, 0.6230072587408348, 0.18962791161850712], [0.9920602459326743, 0.6104024959410064, 0.1840886756924244], [0.9917833282948878, 0.5976302231717846, 0.17823961163192695], [0.9912891871817501, 0.584660910964709, 0.17214700759084267], [0.9905631400900912, 0.5714677122588908, 0.16587767290499006], [0.9895905045167414, 0.5580263006318681, 0.15949881685675113], [0.9883565979585309, 0.5443147998853488, 0.15307790816828803], [0.9868467379122904, 0.5303138044684916, 0.14668251522340267], [0.9850462418748495, 0.5160064896870945, 0.14038012701804065], [0.9829415325780593, 0.5013871258172927, 0.13423136665239896], [0.9805283806463996, 0.48652287501129976, 0.12824076784310964], [0.9778072570671092, 0.4715174832781621, 0.12238470698953774], [0.9747786684299458, 0.45647541200570035, 0.11663935572331047], [0.9714431213246664, 0.44150112246113676, 0.11098098205678426], [0.9678011223410284, 0.42669869332027743, 0.10538603876578982], [0.9638531780687892, 0.4121715013388048, 0.09983125148265794], [0.9595997950977064, 0.3980219636495409, 0.09429370649952155], [0.955041480017537, 0.38435133958413126, 0.08875093828190032], [0.9501776477067925, 0.3712391645206384, 0.08319244458709686], [0.945002869918086, 0.35867340216680627, 0.07765856363342656], [0.939510375845868, 0.34661657642435645, 0.07220291355950477], [0.9336933946348576, 0.33503080017448555, 0.06687804720630672], [0.927545155429773, 0.3238776561046621, 0.061735278666316916], [0.9210588873753326, 0.3131181839572251, 0.056824503280264636], [0.9142278196162553, 0.30271297234473005, 0.05219401055843878], [0.9070451812972594, 0.2926223514744295, 0.04789029002658525], [0.8995042852162572, 0.2828067015124927, 0.043957748795836256], [0.891624412799365, 0.27323253076797877, 0.04041351972303193], [0.8834874722688015, 0.263879251265467, 0.03721289045871814], [0.8751846232207746, 0.254725704246588, 0.03430224569442672], [0.8668070252514929, 0.24574829852791777, 0.03162849539198234], [0.8584458379571641, 0.23692126221807738, 0.029139373140852975], [0.8501922209339968, 0.22821677001517918, 0.026783627401313844], [0.842137333778199, 0.2196049425900127, 0.024511105633431018], [0.8343723360859792, 0.21105371449732208, 0.022272731311857318], [0.8269868884473934, 0.20252933155631014, 0.02002124877275106], [0.8200045690445795, 0.19402936800454978, 0.017748501028958072], [0.8133583005373909, 0.18559748229672426, 0.015498635760688948], [0.8069744374385174, 0.17728209148881974, 0.013318655178400935], [0.8007793342606484, 0.16913337148852742, 0.0112547441788015], [0.7946993455164736, 0.16120340729781465, 0.009352359692418623], [0.788660825718683, 0.15354617809025697, 0.00765622813291652], [0.782590129379966, 0.14621737391166023, 0.0062102509481560386], [0.7764136110130129, 0.1392740409365317, 0.005057318272995783], [0.7700620750740421, 0.13276939181212588, 0.004236488419270876], [0.7635366882039091, 0.12668307615009333, 0.003745345772828322], [0.7568947816167044, 0.12093573368910132, 0.003549296943119582], [0.7501952516603297, 0.11544669842075488, 0.0036134540477882945], [0.7434969946826867, 0.11013619050381865, 0.0039037799561516488], [0.7368589070316779, 0.10492567700866447, 0.004387290052200903], [0.7303398850552048, 0.09973804029508934, 0.0050322105475879364], [0.7239988251011695, 0.09449755619814312, 0.005808093344599986], [0.7178946235174741, 0.08912968460872006, 0.006685887449122245], [0.7120861766520206, 0.08356067512355873, 0.007637966933587637], [0.7066323808527106, 0.07771699020297941, 0.008638115449914247], [0.7015921324674467, 0.07152454771412878, 0.009661467292431428], [0.6970243278441304, 0.06490778385274144, 0.01068440501079227], [0.6929878633306636, 0.05778853622743917, 0.011684413572873753], [0.6895416352749486, 0.050084745357375754, 0.012639891077665865], [0.6867445400248873, 0.041708970976610964, 0.013529916018146956], [0.6846554739283816, 0.032566717356936344, 0.014333971094147597], [0.6833333333333333, 0.022554559355018072, 0.01503162357520156], [0.4796, 0.01583, 0.010550000000000002]]


root_directory = os.path.dirname(os.path.realpath(__file__))

np.random.seed(0)

plt.close('all')


def sample_multimodal_distribution(size):
    
    randseeds   = np.random.uniform(size=size)
    
    seeds1  = np.where(np.logical_and(
        randseeds >= 0,
        randseeds < 1/3))
    
    seeds2  = np.where(np.logical_and(
        randseeds >= 1/3,
        randseeds < 2/3))
    
    seeds3  = np.where(randseeds >= 2/3)
    
    X   = np.zeros((size,2))
    
    
    color   = [[]]*size
    
    
    color1      = np.asarray(colors.to_rgb("xkcd:goldenrod"))
    color2      = np.asarray(colors.to_rgb("xkcd:orangish red"))
    
    X[seeds1,:] = scipy.stats.multivariate_normal.rvs(
        mean    = [-1.5,-1.5],
        cov     = np.identity(2)*0.1,
        size    = len(seeds1[0]))
    
    for idx in seeds1[0]:
        dist        = np.sqrt(np.sum((X[idx,:] - np.array([-1.5,-1.5]))**2))
        color[idx]  = color2 + (color1 - color2)*scipy.stats.norm.pdf(x=dist,scale=np.sqrt(0.1))/scipy.stats.norm.pdf(x=0,scale=np.sqrt(0.1))
    
    color1      = np.asarray(colors.to_rgb("xkcd:yellow green"))
    color2      = np.asarray(colors.to_rgb("xkcd:grass green"))
    
    X[seeds2,:] = scipy.stats.multivariate_normal.rvs(
        mean    = [0,1.5],
        cov     = np.identity(2)*0.1,
        size    = len(seeds2[0]))
    
    for idx in seeds2[0]:
        dist        = np.sqrt(np.sum((X[idx,:] - np.array([0,1.5]))**2))
        color[idx]  = color2 + (color1 - color2)*scipy.stats.norm.pdf(x=dist,scale=np.sqrt(0.1))/scipy.stats.norm.pdf(x=0,scale=np.sqrt(0.1))
    
    color1      = np.asarray(colors.to_rgb("xkcd:sky blue"))
    color2      = np.asarray(colors.to_rgb("xkcd:cerulean"))
    
    X[seeds3,:] = scipy.stats.multivariate_normal.rvs(
        mean    = [1.5,0],
        cov     = np.identity(2)*0.1,
        size    = len(seeds3[0]))
    
    for idx in seeds3[0]:
        dist        = np.sqrt(np.sum((X[idx,:] - np.array([1.5,0]))**2))
        color[idx]  = color2 + (color1 - color2)*scipy.stats.norm.pdf(x=dist,scale=np.sqrt(0.1))/scipy.stats.norm.pdf(x=0,scale=np.sqrt(0.1))
        
    color   = np.asarray(color)
    
    return X, color

def evaluate_density_multimodal_distribution(X):
    
    density     = scipy.stats.multivariate_normal.pdf(
        x       = X,
        mean    = [-1.5,-1.5],
        cov     = np.identity(2)*0.1)
    
    density     += scipy.stats.multivariate_normal.pdf(
        x       = X,
        mean    = [0,1.5],
        cov     = np.identity(2)*0.1)
    
    density     += scipy.stats.multivariate_normal.pdf(
        x       = X,
        mean    = [1.5,0],
        cov     = np.identity(2)*0.1)
    
    density     /= 3
    
    return density

def sample_spiral_distribution(size):
    
    # First draw some rotation samples from a beta distribution, then scale 
    # them to the range between -pi and +2pi
    seeds = scipy.stats.beta.rvs(
        a       = 4,
        b       = 3,
        size    = size)*3*np.pi-np.pi
    
    # Create a local copy of the rotations
    seeds_orig = copy.copy(seeds)
    
    # Re-normalize the rotations, then scale them to the range between [-3,+3]
    vals    = (seeds+np.pi)/(3*np.pi)*6-3
    
    # Plot the rotation samples on a straight spiral
    X       = np.column_stack((
        np.cos(seeds)[:,np.newaxis],
        np.sin(seeds)[:,np.newaxis]))*((1+seeds+np.pi)/(3*np.pi)*5)[:,np.newaxis]

    # Offset each sample along the spiral's normal vector by scaled Gaussian 
    # noise
    X   += np.column_stack([
        np.cos(seeds_orig),
        np.sin(seeds_orig)])*(scipy.stats.norm.rvs(size=size)*scipy.stats.norm.pdf(vals))[:,np.newaxis]

    return X/2

def sample_wavy_distribution(size):
    
    # First draw some rotation samples from a beta distribution, then scale 
    # them to the range between -pi and +2pi
    seeds = scipy.stats.beta.rvs(
        a       = 2,
        b       = 2,
        size    = size)*2*np.pi-1*np.pi
    
    # Plot the rotation samples on a straight spiral
    X       = np.column_stack((
        seeds[:,np.newaxis],
        np.sin(seeds)[:,np.newaxis]))#*((1+seeds+np.pi)/(3*np.pi)*5)[:,np.newaxis]
    
    # Offset each sample along the spiral's normal vector by scaled Gaussian 
    # noise
    # Normal is (-dy,dx)
    # (dx,dy) of sin curve is (1,np.cos(seeds))
    X   += np.column_stack([
        -np.cos(seeds)/np.sqrt(np.cos(seeds)**2 + 1),
        np.sqrt(np.cos(seeds)**2 + 1)])*(scipy.stats.norm.rvs(scale=0.3,size=size)*scipy.stats.norm.pdf(seeds,scale=2))[:,np.newaxis]
    
    X[:,0] /= 1.5
    X[:,1] *= 1.5
        
    return X

# Training ensemble size
N   = 10000

# Draw that many samples
X, color = sample_multimodal_distribution(N)


# =============================================================================
# Train the transport map
# =============================================================================

# Let's see if tm can map standard Gaussian samples to the target
norm_samples    = scipy.stats.norm.rvs(size=(10000,2))

# Get the directional color values for the Gaussian samples
color_values_directional = []
for i in range(norm_samples.shape[0]):
    
    # Convert their location relative to the origin to an angle
    dummy   = math.atan2(norm_samples[i,1],norm_samples[i,0])
    
    # # Then extract the corresponding colour on a circular colormap (hsv)
    # dummy   = list(matplotlib.cm.hsv((dummy+np.pi)/(2*np.pi)))
    
    dummy   = turbocolors[int(np.round((dummy+np.pi)/(2*np.pi)*255))]
    
    # And write them into the color list
    color_values_directional.append(dummy)
    

# Convert the list into an array
color_values_directional    = np.asarray(color_values_directional)

# Create a slightly different meshgrid
x,y = np.meshgrid(
    np.linspace(-3,3,101),
    np.linspace(-3,3,101))

# Extract the target distribution's density from the Kernel Density Estimator
xy  = np.column_stack((
    np.ndarray.flatten(x),
    np.ndarray.flatten(y)))
z   = evaluate_density_multimodal_distribution(xy)
z   = z.reshape((101,101))
    
z_ref   = scipy.stats.multivariate_normal.pdf(
    xy,
    mean    = np.zeros(2),
    cov     = np.identity(2))

z_ref   = z_ref.reshape((101,101))

resolution  = 31
binpos      = np.linspace(-3,3,resolution+1)[1:]
binpos      -= (binpos[1]-binpos[0])/2
binwidth    = binpos[1]-binpos[0]
bins        = [[x-binwidth/2,x+binwidth/2] for x in binpos]

order   = 10

#%%

plt.close('all')
plt.figure(figsize=(12,7)) # 14,21

gs_supra = matplotlib.gridspec.GridSpec(
    nrows           = 2,
    ncols           = 2,
    wspace          = 0.15,
    hspace          = 0.3)

for oi,order in enumerate([1,10]):

    
    # Create empty lists for the map component specifications
    monotone    = []
    nonmonotone = []
    
    # Specify the map components
    for k in range(2):
        
        # Level 1: One list entry for each dimension (two in total)
        monotone.append([])
        nonmonotone.append([[]]) # An empty list "[]" denotes a constant
    
        # Go through each polynomial order
        for o in range(order):
            
            # Nonmonotone part ----------------------------------------------------
            
            if k > 0:
                # Level 2: Specify the polynomial order of this term;
                # It's a Hermite function term
                if o == 0:
                    nonmonotone[-1].append([k-1]*(o+1))
                else:
                    nonmonotone[-1].append([k-1]*(o+1)+['HF'])
            
            # Monotone part -------------------------------------------------------
            
            # For the monotone part, we consider cross-terms. This list creates all
            # possible combinations for this map component up to total order 10
            comblist = list(itertools.combinations_with_replacement(
                np.arange(k+1),#[-1:],
                o+1))
            
            # Go through each of these combinations
            for entry in comblist:
                
                # Create a transport map component term
                if k in entry:
                    
                    if entry == (k,):
                        
                        if order == 1:
                            monotone[-1].append(
                                [])
                        else:
                            monotone[-1].append(
                                [])
                            monotone[-1].append(
                                list(entry)+['HF'])
                        
                    else:
                    
                        # Level 2: Specify the polynomial order of this term;
                        # It's a Hermite function
                        monotone[-1].append(
                            list(entry)+['HF'])
                        
    # raise Exception
                                    
    # Delete any map object which might already exist
    if "tm" in globals():
        del tm
        
    # Parameterize the transport map
    tm     = transport_map(
        monotone                = monotone,
        nonmonotone             = nonmonotone,
        X                       = copy.copy(X),         # Training ensemble
        polynomial_type         = "hermite function",   # We use Hermite functions for stability
        monotonicity            = "integrated rectifier", # Because we have cross-terms, we require the integrated rectifier formulation
        standardize_samples     = True,                 # Standardize X before training
        workers                 = 1,                    # Number of workers for the parallel optimization; 1 is not parallel
        quadrature_input        = {                     # Keywords for the Gaussian quadrature used for integration
            'order'         : 25,       # If the map is bad, increase this number; takes more computational effort
            'adaptive'      : False,
            'threshold'     : 1E-9,
            'verbose'       : False,
            'increment'     : 6})
    
    # Train the map
    if 'dict_coeffs_order='+str(order)+'.p' not in os.listdir(root_directory):
        
        print('No pre-optimized coefficients found. Optimizing...')
        
        tm.optimize()
    
        dict_coeffs = {
            'coeffs_mon'    : tm.coeffs_mon,
            'coeffs_nonmon' : tm.coeffs_nonmon}
        pickle.dump(dict_coeffs,open('dict_coeffs_order='+str(order)+'.p','wb'))
        
    else:
        
        print('Pre-optimized coefficients found. Extracting...')
        
        dict_coeffs = pickle.load(open('dict_coeffs_order='+str(order)+'.p','rb'))
        
        tm.coeffs_mon       = copy.copy(dict_coeffs['coeffs_mon'])
        tm.coeffs_nonmon    = copy.copy(dict_coeffs['coeffs_nonmon'])
        
    # Apply the inverse map
    ret = tm.inverse_map(
        Z           = copy.copy(norm_samples))
    
    
    pushforward = tm.map(X)
    pullback = tm.inverse_map(copy.copy(norm_samples))
    
    # Get the directional color values for the Gaussian samples
    color_values_directional = []
    for i in range(norm_samples.shape[0]):
        
        # Convert their location relative to the origin to an angle
        dummy   = math.atan2(norm_samples[i,1],norm_samples[i,0])
        
        dummy   = turbocolors[int(np.round((dummy+np.pi)/(2*np.pi)*255))]
        
        # And write them into the color list
        color_values_directional.append(dummy)
        
    
    # Convert the list into an array
    color_values_directional_reference    = np.asarray(color_values_directional)
    
    color_values_directional    = color
    
    # =========================================================================
    # Show the forward mao
    # =========================================================================
    
    def coords_to_axes_fraction(x):
        
        return (x + 3)/6

    #%%

    # Plot the first samples
    gs_mid = gridspec.GridSpecFromSubplotSpec(
        nrows           = 1,
        ncols           = 2,
        wspace          = 0.1, 
        hspace          = 0.0,
        subplot_spec    = gs_supra[0,oi])
    
    
    gs  = matplotlib.gridspec.GridSpecFromSubplotSpec(
        nrows           = 2,
        ncols           = 2,
        width_ratios    = [0.2,1.0],
        height_ratios   = [1.,0.2],
        wspace          = 0.0, 
        hspace          = 0.0,
        subplot_spec    = gs_mid[0,0])
    
    plt.subplot(gs[0,1])
    
    plt.scatter(X[::10,0],X[::10,1],facecolor = color_values_directional[::10], edgecolor = "None", zorder = 100, s = 10)

    plt.gca().set_xlim([-3,3])
    plt.gca().set_ylim([-3,3])
    
    # Remove all axis ticks
    plt.tick_params(left=False,
                    bottom=False,
                    labelleft=False,
                    labelbottom=False)
    
    plt.title("target "+"$\pi$", fontsize = 10)
    
    plt.ylabel("forward map")

    # -------------------------------------------------------------------------
    # x1 barplot
    # -------------------------------------------------------------------------
    
    x       = X[::10,0]
    y       = X[::10,1]
    
    plt.subplot(gs[1,1])

    # Remove all axis ticks
    plt.tick_params(left=False,
                    bottom=False,
                    right=False,
                    top=False,
                    labelleft=False,
                    labelbottom=False,
                    labelright=False,
                    labeltop=False)
    
    plt.xlim([-3,3])
    
    plt.gca().invert_yaxis()
    
    # Get the indices in the bin, sort them by their y position
    indices = []
    for ind in [np.where(np.logical_and(x >= bn[0],x < bn[1])) for bn in bins]:
        indices.append(ind[0][np.flip(np.argsort(y[ind]))])
    
    # Plot manually
    for bni,bn in enumerate(bins):
        for j,ind in enumerate(indices[bni]):
            bnred   = binwidth*0.1
            plt.fill(
                [bn[0]+bnred,bn[1]-bnred,bn[1]-bnred,bn[0]+bnred],
                np.asarray([j,j,j+1,j+1]),
                color   = color_values_directional[::10][ind,:],
                ec      = "None",
                zorder  = 10)

    plt.text(
        0.975,
        0.1,
        "$x_{1}$",
        transform = plt.gca().transAxes,
        ha = "right",
        va = "bottom",
        color = "#FF5000")

    # -------------------------------------------------------------------------
    # x2 barplot
    # -------------------------------------------------------------------------
    
    x       = X[::10,0]
    y       = X[::10,1]
    
    plt.subplot(gs[0,0])
    
    # Get the indices in the bin, sort them by their y position
    indices = []
    for ind in [np.where(np.logical_and(y >= bn[0],y < bn[1])) for bn in bins]:
        indices.append(ind[0][np.argsort(x[ind])])
    
    # Plot manually
    for bni,bn in enumerate(bins):
        for j,ind in enumerate(indices[bni]):
            bnred   = binwidth*0.1
            plt.fill(
                [j,j,j+1,j+1],
                [bn[0]+bnred,bn[1]-bnred,bn[1]-bnred,bn[0]+bnred],
                color = color_values_directional[::10][ind,:],
                ec      = "None",
                zorder  = 10)
    
            
    plt.ylim([-3,3])
    
    plt.gca().invert_xaxis()
    
    # Remove all axis ticks
    plt.tick_params(left=False,
                    bottom=False,
                    right=False,
                    top=False,
                    labelleft=False,
                    labelbottom=False,
                    labelright=False,
                    labeltop=False)
    
    plt.text(
        0.1,
        0.025,
        "$x_{2}$",
        transform = plt.gca().transAxes,
        ha = "left",
        va = "bottom",
        color = "#FF5000")

    #%%

    # Plot the first samples
    
    gs  = matplotlib.gridspec.GridSpecFromSubplotSpec(
        nrows           = 2,
        ncols           = 2,
        width_ratios    = [1.,0.2],
        height_ratios   = [1.,0.2],
        wspace          = 0.0, 
        hspace          = 0.0,
        subplot_spec    = gs_mid[0,1])

    plt.subplot(gs[0,0])
    
    
    plt.gca().annotate('', xy=(0.17, 0.5), xycoords='axes fraction', xytext=(0.17-0.001, 0.5), 
                        arrowprops=dict(color="xkcd:dark grey",headlength=20,headwidth=35,width=1),)
    
    for i in range(25):
        
        plt.gca().annotate('', xy=(-i/25*0.125, 0.41), xycoords='axes fraction', xytext=(-i/25*0.125, 0.59), 
                arrowprops=dict(color="xkcd:dark grey",arrowstyle="-",lw=2),)
        
    plt.text(
        -0.025,
        0.5,
        r"$\mathbf{S}$",
        transform = plt.gca().transAxes,
        color = "w",
        ha = "right",
        va = "center")
    
    plt.scatter(pushforward[::10,0],pushforward[::10,1],facecolor = color_values_directional[::10], edgecolor = "None", zorder = 100, s = 10)
    
    plt.title("pushforward "+r"$\mathbf{S}_{\#}\pi$", fontsize = 10)
    
    plt.gca().set_xlim([-3,3])
    plt.gca().set_ylim([-3,3])
    
    # Remove all axis ticks
    plt.tick_params(left=False,
                    bottom=False,
                    labelleft=False,
                    labelbottom=False)
    
    ret_norm_seeds  = tm.inverse_map(
        X_star      = norm_samples[:,0][:,np.newaxis]*0,
        Z           = norm_samples[:,1][:,np.newaxis])[:,1]
    
    if oi == 0:
        
        plt.text(
            -0.05,
            1.2,
            r"$\mathbf{A}:$ simple forward map (linear)",
            transform = plt.gca().transAxes,
            color = "k",
            ha = "center",
            va = "center")
        
    else:
        
        plt.text(
            -0.05,
            1.2,
            r"$\mathbf{C}:$ complex forward map (nonlinear)",
            transform = plt.gca().transAxes,
            color = "k",
            ha = "center",
            va = "center")    
    
    # -------------------------------------------------------------------------
    # x1 barplot
    # -------------------------------------------------------------------------
    
    x       = pushforward[::10,0]
    y       = pushforward[::10,1]
    
    plt.subplot(gs[1,0])

    # Remove all axis ticks
    plt.tick_params(left=False,
                    bottom=False,
                    right=False,
                    top=False,
                    labelleft=False,
                    labelbottom=False,
                    labelright=False,
                    labeltop=False)
    
            
     
    plt.xlim([-3,3])
    
    plt.gca().invert_yaxis()
    
    # Get the indices in the bin, sort them by their y position
    indices = []
    for ind in [np.where(np.logical_and(x >= bn[0],x < bn[1])) for bn in bins]:
        indices.append(ind[0][np.flip(np.argsort(y[ind]))])
    
    # Plot manually
    for bni,bn in enumerate(bins):
        for j,ind in enumerate(indices[bni]):
            bnred   = binwidth*0.1
            plt.fill(
                [bn[0]+bnred,bn[1]-bnred,bn[1]-bnred,bn[0]+bnred],
                np.asarray([j,j,j+1,j+1]),
                color   = color_values_directional[::10][ind,:],
                ec      = "None",
                zorder  = 10)

    plt.text(
        0.975,
        0.1,
        r"$\tilde{z}_{1}$",
        transform = plt.gca().transAxes,
        ha = "right",
        va = "bottom",
        color = "#01A109")

    # -------------------------------------------------------------------------
    # x2 barplot
    # -------------------------------------------------------------------------
    
    x       = pushforward[::10,0]
    y       = pushforward[::10,1]
    
    plt.subplot(gs[0,1])
    
    # Get the indices in the bin, sort them by their y position
    indices = []
    for ind in [np.where(np.logical_and(y >= bn[0],y < bn[1])) for bn in bins]:
        indices.append(ind[0][np.argsort(x[ind])])
    
    # Plot manually
    for bni,bn in enumerate(bins):
        for j,ind in enumerate(indices[bni]):
            bnred   = binwidth*0.1
            plt.fill(
                [j,j,j+1,j+1],
                [bn[0]+bnred,bn[1]-bnred,bn[1]-bnred,bn[0]+bnred],
                color = color_values_directional[::10][ind,:],
                ec      = "None",
                zorder  = 10)
    
            
    plt.ylim([-3,3])
    
    # plt.gca().invert_xaxis()
    
    # Remove all axis ticks
    plt.tick_params(left=False,
                    bottom=False,
                    right=False,
                    top=False,
                    labelleft=False,
                    labelbottom=False,
                    labelright=False,
                    labeltop=False)
    
    plt.text(
        0.9,
        0.025,
        r"$\tilde{z}_{2}$",
        transform = plt.gca().transAxes,
        ha = "right",
        va = "bottom",
        color = "#01A109")
    

    
    #%%
    
    # Plot the first samples
    gs_mid = gridspec.GridSpecFromSubplotSpec(
        nrows           = 1,
        ncols           = 2,
        wspace          = 0.1, 
        hspace          = 0.0,
        subplot_spec    = gs_supra[1,oi])
    
    
    gs  = matplotlib.gridspec.GridSpecFromSubplotSpec(
        nrows           = 2,
        ncols           = 2,
        width_ratios    = [0.2,1.0],
        height_ratios   = [1.,0.2],
        wspace          = 0.0, 
        hspace          = 0.0,
        subplot_spec    = gs_mid[0,0])
    
    plt.subplot(gs[0,1])
    
    plt.scatter(pullback[::10,0],pullback[::10,1],facecolor = color_values_directional_reference[::10], edgecolor = "None", zorder = 100, s = 10)

    

    plt.gca().set_xlim([-3,3])
    plt.gca().set_ylim([-3,3])
    
    # Remove all axis ticks
    plt.tick_params(left=False,
                    bottom=False,
                    labelleft=False,
                    labelbottom=False)
    
    plt.title("pullback "+r"$\mathbf{S}^{\#}\eta$", fontsize = 10)
    
    # -------------------------------------------------------------------------
    # x1 barplot
    # -------------------------------------------------------------------------
    
    x       = pullback[::10,0]
    y       = pullback[::10,1]
    
    plt.subplot(gs[1,1])

    # Remove all axis ticks
    plt.tick_params(left=False,
                    bottom=False,
                    right=False,
                    top=False,
                    labelleft=False,
                    labelbottom=False,
                    labelright=False,
                    labeltop=False)
    
            
     
    plt.xlim([-3,3])
    
    plt.gca().invert_yaxis()
    
    # Get the indices in the bin, sort them by their y position
    indices = []
    for ind in [np.where(np.logical_and(x >= bn[0],x < bn[1])) for bn in bins]:
        indices.append(ind[0][np.flip(np.argsort(y[ind]))])
    
    # Plot manually
    for bni,bn in enumerate(bins):
        for j,ind in enumerate(indices[bni]):
            bnred   = binwidth*0.1
            plt.fill(
                [bn[0]+bnred,bn[1]-bnred,bn[1]-bnred,bn[0]+bnred],
                np.asarray([j,j,j+1,j+1]),
                color   = color_values_directional_reference[::10][ind,:],
                ec      = "None",
                zorder  = 10)

    plt.text(
        0.975,
        0.1,
        "$x_{1}$",
        transform = plt.gca().transAxes,
        ha = "right",
        va = "bottom",
        color = "#FF5000")
    
    # -------------------------------------------------------------------------
    # x2 barplot
    # -------------------------------------------------------------------------
    
    x       = pullback[::10,0]
    y       = pullback[::10,1]
    
    plt.subplot(gs[0,0])
    
    # Get the indices in the bin, sort them by their y position
    indices = []
    for ind in [np.where(np.logical_and(y >= bn[0],y < bn[1])) for bn in bins]:
        indices.append(ind[0][np.argsort(x[ind])])
    
    # Plot manually
    for bni,bn in enumerate(bins):
        for j,ind in enumerate(indices[bni]):
            bnred   = binwidth*0.1
            plt.fill(
                [j,j,j+1,j+1],
                [bn[0]+bnred,bn[1]-bnred,bn[1]-bnred,bn[0]+bnred],
                color = color_values_directional_reference[::10][ind,:],
                ec      = "None",
                zorder  = 10)
    
            
    plt.ylim([-3,3])
    
    plt.gca().invert_xaxis()
    
    # Remove all axis ticks
    plt.tick_params(left=False,
                    bottom=False,
                    right=False,
                    top=False,
                    labelleft=False,
                    labelbottom=False,
                    labelright=False,
                    labeltop=False)
    
    plt.text(
        0.1,
        0.025,
        "$x_{2}$",
        transform = plt.gca().transAxes,
        ha = "left",
        va = "bottom",
        color = "#FF5000")

    #%%

    # Plot the first samples
    gs  = matplotlib.gridspec.GridSpecFromSubplotSpec(
        nrows           = 2,
        ncols           = 2,
        width_ratios    = [1.,0.2],
        height_ratios   = [1.,0.2],
        wspace          = 0.0, 
        hspace          = 0.0,
        subplot_spec    = gs_mid[0,1])

    plt.subplot(gs[0,0])
    
    plt.gca().annotate('', xy=(-0.28, 0.5), xycoords='axes fraction', xytext=(-0.28+0.001, 0.5), 
                        arrowprops=dict(color="xkcd:dark grey",headlength=20,headwidth=35,width=1),)
    
    for i in range(25):
        
        plt.gca().annotate('', xy=(-i/25*0.125, 0.41), xycoords='axes fraction', xytext=(-i/25*0.125, 0.59), 
                arrowprops=dict(color="xkcd:dark grey",arrowstyle="-",lw=2),)
        
    plt.text(
        -0.025,
        0.5,
        r"$\mathbf{S}^{-1}$",
        transform = plt.gca().transAxes,
        color = "w",
        ha = "right",
        va = "center")
    
    plt.scatter(norm_samples[::10,0],norm_samples[::10,1],facecolor = color_values_directional_reference[::10], edgecolor = "None", zorder = 100, s = 10)
    
    plt.title("reference "+r"$\eta$", fontsize = 10)
    
    plt.gca().set_xlim([-3,3])
    plt.gca().set_ylim([-3,3])
    
    # Remove all axis ticks
    plt.tick_params(left=False,
                    bottom=False,
                    labelleft=False,
                    labelbottom=False)
    
    if oi == 0:
        
        plt.text(
            -0.05,
            1.2,
            r"$\mathbf{B}:$ simple inverse map (linear)",
            transform = plt.gca().transAxes,
            color = "k",
            ha = "center",
            va = "center")
        
    else:
        
        plt.text(
            -0.05,
            1.2,
            r"$\mathbf{D}:$ complex inverse map (nonlinear)",
            transform = plt.gca().transAxes,
            color = "k",
            ha = "center",
            va = "center")    
    
    # -------------------------------------------------------------------------
    # x1 barplot
    # -------------------------------------------------------------------------
    
    x       = norm_samples[::10,0]
    y       = norm_samples[::10,1]
    
    plt.subplot(gs[1,0])

    # Remove all axis ticks
    plt.tick_params(left=False,
                    bottom=False,
                    right=False,
                    top=False,
                    labelleft=False,
                    labelbottom=False,
                    labelright=False,
                    labeltop=False)
    
    plt.xlim([-3,3])
    
    plt.gca().invert_yaxis()
    
    # Get the indices in the bin, sort them by their y position
    indices = []
    for ind in [np.where(np.logical_and(x >= bn[0],x < bn[1])) for bn in bins]:
        indices.append(ind[0][np.flip(np.argsort(y[ind]))])
    
    # Plot manually
    for bni,bn in enumerate(bins):
        for j,ind in enumerate(indices[bni]):
            bnred   = binwidth*0.1
            plt.fill(
                [bn[0]+bnred,bn[1]-bnred,bn[1]-bnred,bn[0]+bnred],
                np.asarray([j,j,j+1,j+1]),
                color   = color_values_directional_reference[::10][ind,:],
                ec      = "None",
                zorder  = 10)

    plt.text(
        0.975,
        0.1,
        r"$z_{1}$",
        transform = plt.gca().transAxes,
        ha = "right",
        va = "bottom",
        color = "#01A109")

    # -------------------------------------------------------------------------
    # x2 barplot
    # -------------------------------------------------------------------------
    
    x       = norm_samples[::10,0]
    y       = norm_samples[::10,1]
    
    plt.subplot(gs[0,1])
    
    # Get the indices in the bin, sort them by their y position
    indices = []
    for ind in [np.where(np.logical_and(y >= bn[0],y < bn[1])) for bn in bins]:
        indices.append(ind[0][np.argsort(x[ind])])
    
    # Plot manually
    for bni,bn in enumerate(bins):
        for j,ind in enumerate(indices[bni]):
            bnred   = binwidth*0.1
            plt.fill(
                [j,j,j+1,j+1],
                [bn[0]+bnred,bn[1]-bnred,bn[1]-bnred,bn[0]+bnred],
                color = color_values_directional_reference[::10][ind,:],
                ec      = "None",
                zorder  = 10)
    
            
    plt.ylim([-3,3])
    
    # Remove all axis ticks
    plt.tick_params(left=False,
                    bottom=False,
                    right=False,
                    top=False,
                    labelleft=False,
                    labelbottom=False,
                    labelright=False,
                    labeltop=False)
    
    plt.text(
        0.9,
        0.025,
        r"$z_{2}$",
        transform = plt.gca().transAxes,
        ha = "right",
        va = "bottom",
        color = "#01A109")
    
plt.gcf().subplots_adjust(top=0.8)
 
plt.savefig("composite_01.png",dpi=300,bbox_inches="tight")
plt.savefig("composite_01.pdf",dpi=300,bbox_inches="tight")
