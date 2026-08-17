/* 解析逻辑单测：验证视觉模型返回 → 4 字段的健壮性（无需 key） */
const { extractWords, parseOcrText } = require('./server.js');

let pass = 0, fail = 0;
function eq(name, got, exp) {
  const g = JSON.stringify(got), e = JSON.stringify(exp);
  if (g === e) { pass++; console.log('  ✓ ' + name); }
  else { fail++; console.log('  ✗ ' + name + '\n    期望 ' + e + '\n    得到 ' + g); }
}

console.log('extractWords:');
eq('纯净 JSON 数组', extractWords('[{"en":"next","phonetic":"nekst","pos":"adj.","zh":"下一个","ex":"Next one, please."}]'),
  [{ en: 'next', phonetic: 'nekst', pos: 'adj.', zh: '下一个', ex: 'Next one, please.' }]);
eq('带 ```json 围栏',
  extractWords('```json\n[{"en":"apple","phonetic":"ˈæp.l","pos":"n.","zh":"苹果"}]\n```'),
  [{ en: 'apple', phonetic: 'ˈæp.l', pos: 'n.', zh: '苹果', ex: '' }]);
eq('带解释文字包裹',
  extractWords('好的，识别结果如下：\n[{"en":"book","phonetic":"bʊk","pos":"n.","zh":"书"}]\n如有问题请修改。'),
  [{ en: 'book', phonetic: 'bʊk', pos: 'n.', zh: '书', ex: '' }]);
eq('缺字段补全',
  extractWords('[{"en":"cat","zh":"猫"}]'),
  [{ en: 'cat', phonetic: '', pos: '', zh: '猫', ex: '' }]);
eq('含斜杠音标被清洗',
  extractWords('[{"en":"thank","phonetic":"/θæŋk/","pos":"v.","zh":"感谢"}]'),
  [{ en: 'thank', phonetic: 'θæŋk', pos: 'v.', zh: '感谢', ex: '' }]);

console.log('parseOcrText (无 key 兜底):');
eq('en /ph/ pos. zh 行',
  parseOcrText('next /nekst/ adj. 下一个'),
  [{ en: 'next', phonetic: '', pos: '', zh: '下一个', ex: '' }]);
eq('多行',
  parseOcrText('apple 苹果\nbanana 香蕉'),
  [{ en: 'apple', phonetic: '', pos: '', zh: '苹果', ex: '' }, { en: 'banana', phonetic: '', pos: '', zh: '香蕉', ex: '' }]);

console.log('\n结果：' + pass + ' 通过，' + fail + ' 失败');
process.exit(fail ? 1 : 0);
