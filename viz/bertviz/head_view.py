import json
from IPython.core.display import display, HTML, Javascript
import os
from .util import format_special_chars, format_attention


def head_view(attn_data, tokens, sentence_b_start = None, prettify_tokens=False):
    """Render head view

        Args:
            attention: list of ``torch.FloatTensor``(one for each layer) of shape
                ``(batch_size(must be 1), num_heads, sequence_length, sequence_length)``
            tokens: list of tokens
            sentence_b_index: index of first wordpiece in sentence B if input text is sentence pair (optional)
            prettify_tokens: indicates whether to remove special characters in wordpieces, e.g. Ġ
    """

    if sentence_b_start is not None:
        vis_html = """
        <span style="user-select:none">
            Layer: <select id="layer"></select>
            Attention: <select id="filter">
              <option value="all">All</option>
              <option value="ab">Text -> RoIs</option>
              <option value="ba">RoIs -> Text</option>
              <option value="aa">Text -> Text</option>
              <option value="bb">RoIs -> RoIs</option>
            </select>
            </span>
        <div id='vis'></div>
        """
    else:
        vis_html = """
              <span style="user-select:none">
                Layer: <select id="layer"></select>
              </span>
              <div id='vis'></div> 
            """

    display(HTML(vis_html))
    __location__ = os.path.realpath(
        os.path.join(os.getcwd(), os.path.dirname(__file__)))
    vis_js = open(os.path.join(__location__, 'head_view.js')).read()

    if prettify_tokens:
        tokens = format_special_chars(tokens)

    params = {
        'attention': attn_data,
        'default_filter': "ab"
    }

    display(Javascript('window.params = %s' % json.dumps(params)))
    display(Javascript(vis_js))
