#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import simplejson as json
from sklearn.metrics import confusion_matrix, classification_report
import click

FOO_CSS = """
body {
    padding-top: 10px;
    padding-bottom: 20px;
}
.story {
    font-size: 80%;
    position: relative;
}
a.popuped {
    text-decoration: none;
}
a.popuped span.popup {
    display: none;
}
a.popuped:hover span.popup {
    position: absolute;
    background: #333333;
    border: #ffffff;
    border-radius: 4px;
    color: #ffffff;
    display: block;
    line-height: 1.1em;
    margin: 0.2em;
    padding: 0.2em;
    width: 5em;
    text-align: center;
}
table.dataframe {
    margin: 0 auto;
    border-radius: 10px;
    -webkit-border-radius: 10px;
    -moz-border-radius: 10px;
    border: 1px solid #666;
    border-spacing: 0;
    overflow: hidden;
}
table.dataframe td, th {
    border-bottom: 1px solid #666;
    padding: 5px 10px;
    text-align: center;
}
table.dataframe th {
    background: #efefef;
}
table.dataframe tbody tr:last-child th,
table.dataframe tbody tr:last-child td {
    border-bottom: none;
}
table.dataframe th + th, td {
    border-left: 1px solid #666;
}
"""

FOO_HTML_HEAD = """<!DOCTYPE html>
<html lang=ja>
<head>
<meta http-equiv="Content-Type" content="text/html; charset=utf-8"/>
<meta name="viewport" content="width=device-width,
      initial-scale=1, shrink-to-fit=no">
<title>{www_path}</title>
<!-- CSS -->
<link href="/font-awesome4/css/font-awesome.min.css" rel="stylesheet">
<link href="/footable/css/footable.standalone.min.css" rel="stylesheet">
<link href="/footable/css/footable.fontawesome.css" rel="stylesheet">
<link href="popup.css" rel="stylesheet">
<!-- JavaScript -->
<script src="/jquery/jquery-3.5.0.min.js" type="text/javascript"></script>
<script src="/footable/js/footable.min.js" type="text/javascript"></script>
<script type="text/javascript">
jQuery(function($){{
    $('.sentence_table').footable({{
        "columns": $.get('/{www_path}/sentence_columns.json'),
        "rows": $.get('/{www_path}/sentence_rows.json'),
        "showToggle": true,
        "toggleColumn": "last",
        "expandFirst": false,
        "filtering": {{
            "enabled": true,
            "dropdownTitle": "Search in:"
        }},
        "paging": {{
            "enabled": true,
            "size": 10
        }},
        "sorting": {{
            "enabled": true
        }}
    }});
}});
</script>
<body>
<h2>{www_path}</h2>
"""
FOO_HTML_BODY = """
<h3>TextCNNによる文書カテゴリ推定とGradCAMによる推論の可視化</h3>
<table class="sentence_table" border="1"></table>
</body>
</html>
"""


def write_columns(html_dir: str) -> None:
    sentence_columns = list()
    sentence_columns.append({
        'name': 'id',
        'title': '番号',
        'type': 'number',
        'filterable': False,
        'sortable': True,
    })
    sentence_columns.append({
        'name': 'true',
        'title': '実際のカテゴリ',
        'type': 'text',
        'filterable': True,
        'sortable': True,
    })
    sentence_columns.append({
        'name': 'pred',
        'title': '予測したカテゴリ',
        'type': 'text',
        'filterable': True,
        'sortable': True,
    })
    sentence_columns.append({
        'name': 'prob',
        'title': '確率',
        'type': 'number',
        'filterable': False,
        'sortable': True,
    })
    sentence_columns.append({
        'name': 'open',
        'title': '記事冒頭（クリックして全文表示切替）',
        'type': 'text',
        'filterable': False,
        'sortable': False,
    })
    sentence_columns.append({
        'name': 'story',
        'title': '記事全文と各単語の重要度',
        'type': 'html',
        'breakpoints': 'all',
        'filterable': False,
        'sortable': False,
    })
    with open(os.path.join(html_dir, "sentence_columns.json"), "wt") as wf:
        json.dump(sentence_columns, wf, ensure_ascii=False, indent=0)
    return


@click.command()
@click.option('--input-json', '-i', type=str, required=True)
@click.option('--www-path', '-w', type=str, required=True)
def main(input_json: str, www_path: str) -> None:
    html_dir = os.path.join('/usr/local/var/www', www_path)
    os.makedirs(html_dir, exist_ok=True)
    with open(os.path.join(html_dir, 'popup.css'), 'wt') as wf:
        wf.write(FOO_CSS)
    write_columns(html_dir)
    with open(input_json, "rt") as rf:
        data = json.load(rf)
    html = ''
    trues = list()
    preds = list()
    rows = list()
    for i, d in enumerate(data):
        row = dict()
        row['id'] = i + 1
        row['true'] = d['category']
        trues.append(d['category'])
        row['pred'] = d['predicted_category']
        preds.append(d['predicted_category'])
        row['prob'] = d['probability']
        row['open'] = d['document'][:30]
        row['story'] = '<div class="story">'
        if 'explain' not in d.keys():
            row['story'] += d['document']
        else:
            row['story'] += "%sカテゴリに対する重要度<br/>" % d['explained_category']
            row['story'] += d['explain']
        row['story'] += '</div>'
        rows.append(row)
    with open(os.path.join(html_dir, 'sentence_rows.json'), 'wt') as wf:
        json.dump(rows, wf, ensure_ascii=True)
    html += FOO_HTML_HEAD.format(www_path=www_path)
    labels = sorted(set(trues + preds))
    cmat = pd.DataFrame(
        confusion_matrix(trues, preds, labels=labels)
    )
    cmat.columns = labels
    cmat.index = labels
    cmat = cmat.assign(実際合計=cmat.sum(axis=1))
    cmat.loc["予測合計"] = cmat.sum(axis=0)
    cmat.fillna(0).astype(np.int)
    html += "<h3>予測精度（Confusion Matrix）</h3>\n"
    html += cmat.to_html() + "\n"
    crep = pd.DataFrame(classification_report(
        trues, preds, zero_division=0, output_dict=True
    )).T
    html += "<h3>予測精度（Accuracy, Precision, Recall）</h3>\n"
    html += crep.to_html() + "\n"
    html += FOO_HTML_BODY
    with open(os.path.join(html_dir, 'index.html'), 'wt') as wf:
        wf.write(html)
    return


if __name__ == "__main__":
    main()
