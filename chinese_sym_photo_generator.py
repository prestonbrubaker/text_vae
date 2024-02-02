from PIL import Image, ImageDraw, ImageFont
import random
import os

FILE_PATH = "photos"

# List of 500 unique Chinese characters
chinese_chars = [
    chinese_chars = [
    '的', '一', '是', '不', '了', '在', '人', '有', '我', '他',
    '这', '个', '们', '中', '来', '上', '大', '为', '和', '国',
    '地', '到', '以', '说', '时', '要', '就', '出', '会', '可',
    '也', '你', '对', '生', '能', '而', '子', '那', '得', '于',
    '着', '下', '自', '之', '年', '过', '发', '后', '作', '里',
    '用', '道', '行', '所', '然', '家', '种', '事', '成', '方',
    '多', '经', '么', '去', '法', '学', '如', '都', '同', '现',
    '当', '没', '动', '面', '起', '看', '定', '天', '分', '还',
    '进', '好', '小', '部', '其', '些', '主', '样', '理', '心',
    '她', '本', '前', '开', '但', '因', '只', '从', '想', '实',
    '日', '军', '者', '意', '无', '力', '它', '与', '长', '把',
    '机', '十', '更', '第', '学', '水', '手', '向', '后', '命',
    '市', '可', '左', '由', '百', '见', '车', '下', '万', '白',
    '东', '字', '路', '口', '头', '女', '太', '明', '点', '文',
    '几', '定', '本', '公', '特', '做', '外', '孩', '相', '西',
    '果', '走', '将', '月', '十', '实', '向', '声', '车', '全',
    '信', '重', '三', '机', '工', '物', '气', '每', '并', '别',
    '真', '打', '夜', '平', '少', '报', '才', '结', '反', '无',
    '开', '很', '但', '因', '只', '从', '让', '问', '很', '最',
    '重', '并', '物', '手', '应', '战', '向', '头', '文', '体',
    '政', '美', '相', '看', '利', '比', '或', '但', '质', '气',
    '第', '向', '道', '命', '此', '变', '条', '只', '没', '结',
    '解', '问', '意', '建', '月', '公', '无', '系', '军', '很',
    '情', '者', '最', '立', '代', '想', '已', '通', '并', '提',
    '直', '题', '党', '程', '展', '五', '果', '料', '象', '员',
    '革', '位', '入', '常', '文', '总', '次', '品', '式', '活',
    '设', '及', '管', '特', '件', '长', '求', '老', '头', '基',
    '资', '边', '流', '路', '级', '少', '图', '山', '统', '接',
    '知', '较', '将', '组', '见', '计', '别', '她', '手', '角',
    '期', '根', '论', '运', '农', '指', '几', '九', '区', '强',
    '放', '决', '西', '被', '干', '做', '必', '战', '先', '回',
    '则', '任', '取', '据', '处', '队', '南', '给', '色', '光',
    '门', '即', '保', '治', '北', '造', '百', '规', '热', '领',
    '七', '海', '口', '东', '导', '器', '压', '志', '世', '金',
    '增', '争', '济', '阶', '油', '思', '术', '极', '交', '受',
    '联', '什', '认', '六', '共', '权', '收', '证', '改', '清',
    '美', '再', '采', '转', '更', '单', '风', '切', '打', '白',
    '教', '速', '花', '带', '安', '场', '身', '车', '例', '真',
    '务', '具', '万', '每', '目', '至', '达', '走', '积', '示',
    '议', '声', '报', '斗', '完', '类', '八', '离', '华', '名',
    '确', '才', '科', '张', '信', '马', '节', '话', '米', '整',
    '空', '元', '况', '今', '集', '温', '传', '土', '许', '步',
    '群', '广', '石', '记', '需', '段', '研', '界', '拉', '林',
    '律', '叫', '且', '究', '观', '越', '织', '装', '影', '算',
    '低', '持', '音', '众', '书', '布', '复', '容', '儿', '须',
    '际', '商', '非', '验', '连', '断', '深', '难', '近', '矿',
    '千', '周', '委', '素', '技', '备', '半', '办', '青', '省',
    '列', '习', '响', '约', '支', '般', '史', '感', '劳', '便',
    '团', '往', '酸', '历', '市', '克', '何', '除', '消', '构',
    '府', '称', '太', '准', '精', '值', '号', '率', '族', '维',
    '划', '选', '标', '写', '存', '候', '毛', '亲', '快', '效',
    '斯', '院', '查', '江', '型', '眼', '王', '按', '格', '养',
    '易', '置', '派', '层', '片', '始', '却', '专', '状', '育',
    '厂', '京', '识', '适', '属', '圆', '包', '火', '住', '调',
    '满', '县', '局', '照', '参', '红', '细', '引', '听', '该',
    '铁', '价', '严', '首', '底', '液', '官', '德', '随', '病',
    '苏', '失', '尔', '死', '讲', '配', '女', '黄', '推', '显',
    '谈', '罪', '神', '艺', '呢', '席', '含', '企', '望', '密',
    '批', '营', '项', '防', '举', '球', '英', '氧', '势', '告',
    '李', '台', '落', '木', '帮', '轮', '破', '亚', '师', '围',
    '注', '远', '字', '材', '排', '供', '河', '态', '封', '另',
    '施', '减', '树', '溶', '怎', '止', '案', '言', '士', '均',
    '武', '固', '叶', '鱼', '波', '视', '仅', '费', '紧', '爱',
    '左', '章', '早', '朝', '害', '续', '轻', '服', '试', '食',
    '充', '兵', '源', '判', '护', '司', '足', '态', '继', '续',
    '棉', '裂', '哥', '速', '忽', '宝', '午', '尘', '闻', '揭',
    '炮', '残', '冬', '桥', '妇', '警', '综', '招', '吴', '付',
    '浮', '遭', '征', '缺', '雨', '吗', '针', '刘', '啊', '急',
    '唱', '误', '训', '愿', '审', '附', '获', '茶', '鲜', '粮',
    '斤', '孩', '脱', '硫', '肥', '善', '龙', '演', '父', '渐',
    '血', '欢', '械', '掌', '歌', '沙', '刚', '攻', '谓', '盾',
    '讨', '晚', '粒', '乱', '燃', '矛', '乎', '杀', '药', '宁',
    '鲁', '贵', '钟', '煤', '读', '班', '伯', '香', '介', '迫',
    '句', '丰', '培', '握', '兰', '担', '弦', '蛋', '沉', '假',
    '穿', '执', '答', '乐', '谁', '顺', '烟', '缩', '征', '脸',
    '喜', '松', '脚', '困', '异', '免', '背', '星', '福', '买',
    '染', '井', '概', '慢', '怕', '磁', '倍', '祖', '皇', '促',
    '静', '补', '评', '翻', '肉', '践', '尼', '衣', '宽', '扬',
    '棉', '希', '伤', '操', '垂', '秋', '宜', '氢', '套', '督',
    '振', '架', '亮', '末', '宪', '庆', '编', '牛', '触', '映',
    '雷', '销', '诗', '座', '居', '抓', '裂', '胞', '呼', '娘',
    '景', '威', '绿', '晶', '厚', '盟', '衡', '鸡', '孙', '延',
    '危', '胶', '屋', '乡', '临', '陆', '顾', '掉', '呀', '灯',
    '岁', '措', '束', '耐', '剧', '玉', '赵', '跳', '哥', '季',
    '课', '凯', '胡', '额', '款', '绍', '卷', '齐', '伟', '蒸',
    '殖', '永', '宗', '苗', '川', '炉', '岩', '弱', '零', '杨',
    '怖', '环', '抱', '柱', '抢', '粉', '泥', '滑', '刻', '痕',
    '迹', '康', '选', '透', '彩', '洲', '励', '层', '弹', '层',
    '翔', '雾', '锋', '藏', '灵', '复', '位', '访', '态', '游',
    '弄', '园', '茂', '英', '典', '壮', '冲', '疯', '炼', '岁',
    '凉', '渡', '悲', '孤', '刊', '丧', '刻', '愁', '刑', '肃',
    '寒', '操', '悔', '课', '宰', '疆', '壤', '炬', '灰', '溜',
    '博', '荒', '喊', '渴', '溃', '辣', '缝', '嫩', '骚', '束',
    '妇', '扮', '涛', '疮', '肿', '豆', '削', '岗', '晃', '吞',
    '宏', '癌', '肚', '隶', '履', '涨', '耀', '扭', '坡', '泼',
    '颖', '霸', '囊', '轰', '咒', '挺', '淋', '潭', '寨', '糕',
    '赌', '塔', '赔', '挪', '聊', '骗', '勃', '宴', '糟', '嘱',
    '豹', '熊', '殷', '翠', '萄', '疯', '淘', '膀', '豪', '腐',
    '撒', '婪', '滚', '桑', '聚', '蛋', '斩', '癸', '霜', '霞',
    '炫', '泼', '镇', '靖', '骤', '魔', '纠', '框', '悠', '曼',
    '赚', '撑', '翘', '蹈', '侠', '滨', '辑', '焰', '渔', '瑟',
    '鹿', '膊', '瑰', '憋', '摊', '搅', '酱', '屏', '疫', '哀',
    '蔡', '堵', '赎', '屡', '躁', '鞭', '懒', '镜', '辜', '嚼',
    '棒', '咕', '膨', '喘', '瘦', '榜', '剥', '凑', '颠', '挫',
    '蹲', '搂', '悟', '慰', '斜', '薪', '靠', '扛', '捣', '胀',
    '蒜', '矮', '辫', '砸', '呜', '疼', '撇', '瘸', '咯', '哄',
    '蜡', '撕', '搁', '蹭', '嘎', '嗓', '喀', '啡', '扒', '巍',
    '咦', '吱', '呕', '啸', '吆', '咏', '嘘', '唉', '哼', '啪',
    '哩', '喽', '嗷', '啦', '叽', '咋', '呐', '喳', '嗖', '嘟',
    '啊', '哇', '喔', '嘻', '嘿', '咚', '嗡', '哒', '噜', '嚓',
    '呻', '喧', '嚷', '嗤', '嚎', '嚏', '嘬', '呲', '哔', '咬',
    '咔', '啄', '嗑', '噎', '咽', '吞', '咆', '嚼', '嗦', '咂',
    '咀', '嚕', '呷', '噘', '咯', '噤', '噢', '咕', '咣', '吁',
    '嗨', '嗯', '啵', '噹', '咻', '呀', '哧', '嘎', '咿', '呓',
]
]

def create_image(image_number):
    global FILE_PATH
    # Create a 256x256 greyscale image with white background
    img = Image.new('L', (256, 256), color=255)

    # Randomly select a Chinese character
    char = random.choice(chinese_chars)

    # Calculate a random font size (20% to 100% of 128 pixels)
    font_size = random.randint(int(0.2 * 128), 128)

    # Load a font that supports Chinese characters
    font = ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", font_size)

    # Create a draw object
    draw = ImageDraw.Draw(img)

    # Calculate the size of the character
    text_width, text_height = draw.textsize(char, font=font)

    # Calculate the position to center the character
    x = (256 - text_width) // 2
    y = (256 - text_height) // 2

    # Draw the character
    draw.text((x, y), char, font=font, fill=0)

    # Save the image in the 'photos' subfolder
    img.save(f'{FILE_PATH}/{image_number:05d}_random_chinese.png')

# Create the 'photos' subfolder if it doesn't exist
if not os.path.exists(FILE_PATH):
    os.makedirs(FILE_PATH)

# Generate and save 100 images
for i in range(100):
    create_image(i)