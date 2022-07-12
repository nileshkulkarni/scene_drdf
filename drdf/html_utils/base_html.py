import json
import os
import os.path as osp
import pdb

import dominate
import dominate.tags as tags

curr_path = osp.dirname(osp.abspath(__file__))
cachedir = osp.join(curr_path, "..", "cachedir")


class SingleHTMLWriter:
    def __init__(self, opts):
        self.opts = opts
        self.web_dir = os.path.join(cachedir, "web")
        print("create web directory %s..." % self.web_dir)
        if opts.ENV_NAME == "main":
            self.env_name = opts.NAME
        else:
            self.env_name = opts.ENV_NAME
            if not (opts.WEB_SUFFIX == ""):
                self.env_name = self.env_name + f"_{opts.WEB_SUFFIX}"

        self.result_dir = osp.join(
            opts.RESULT_DIR, self.env_name, opts.DATALOADER.SPLIT
        )
        return

    def add_data(self, data):
        ## adds all the data for the training step.
        raise NotImplementedError

    def dump_html(self, html_file):
        ## creates a stand alone html and dumps it to the html_file
        raise NotImplementedError


include_html_script = """
  $(function(){
    var includes = $('[data-include]');
    console.log(includes)
    console.log(includes.length)
    includes.each(function(){
      //var file = $(this).data('include') + '.html';;
      var file = '../results/debug_1ex/train/00000001/data' + '.html'
      console.log(file)
    #   console.log(this)
      $(this).load(file);
    });
  });
"""

onclick_script = """
      $(function () {
         toggle = function (elm, base_img, trg_img) {
            // Now the object is $(elm)
            // $(elm).src = ($(elm).src === base_img) ? toggle_img : base_img;
            // console.log(base_img)
            // console.log(elm.src)
            var pieces = elm.src.split('/')
            var filename = pieces[pieces.length - 1];
            var basename_pieces  = base_img.split('/')
            var basename_split = basename_pieces[basename_pieces.length - 1]
            // console.log(filename)
            // console.log((filename === base_img))
            elm.src = (filename === basename_split) ? trg_img : base_img;
         };
      });
   """

lazy_load_script = """
$('.lazyload').lazyload({
  // Sets the pixels to load earlier. Setting threshold to 200 causes image to load 200 pixels
  // before it appears on viewport. It should be greater or equal zero.
  threshold: 200,
  // Sets the callback function when the load event is firing.
  // element: The content in lazyload tag will be returned as a jQuery object.
  load: function(element) {},
  // Sets events to trigger lazyload. Default is customized event `appear`, it will trigger when
  // element appear in screen. You could set other events including each one separated by a space.
  trigger: "appear"
});
"""


class ExpHTMLWriter:
    def __init__(
        self,
        opts,
        step_html_writer,
    ):
        self.step_html_writer = step_html_writer
        self.opts = opts
        self.web_dir = os.path.join(cachedir, "web")
        print("create web directory %s..." % self.web_dir)
        # util.mkdirs([self.web_dir, self.img_dir])
        if opts.ENV_NAME == "main":
            self.env_name = opts.NAME
        else:
            self.env_name = opts.ENV_NAME
        html_name = self.env_name
        os.makedirs(self.web_dir, exist_ok=True)
        self.html_file = osp.join(
            self.web_dir, f"{html_name}_{opts.DATALOADER.SPLIT}.html"
        )
        self.step_htmls = []

    def create_header(self, doc):
        with doc.head:
            tags.script(
                type="text/javascript",
                src="https://cdn.jsdelivr.net/npm/vanilla-lazyload@12.1.0/dist/lazyload.min.js",
            )
            tags.script(
                src="https://ajax.googleapis.com/ajax/libs/jquery/2.1.3/jquery.min.js"
            )
            tags.style(
                """h1 {text-align: center;}
                        p {text-align: center;}
                        div {text-align: center;}"""
            )
            tags.script(lazy_load_script)

        return doc

    def create_footer(self, doc):
        with doc.head:
            # tags.script(include_html_script)
            tags.script(onclick_script)
            tags.script("var LazyLoadInstance = new LazyLoad();")

        return doc

    def write_html(
        self,
    ):
        doc = dominate.document()
        self.create_header(doc)

        with doc.head as doc_head:
            for step_divs in self.step_htmls[::-1]:
                doc_head.add(step_divs)

        self.create_footer(doc)
        with open(self.html_file, "w") as f:
            f.write(doc.render())
        return

    def save_current_results(self, step, visuals):
        self.step_html_writer.add_data(visuals)
        divs = self.step_html_writer.dump(step, self.web_dir)
        self.step_htmls.append(divs)
        self.write_html()
        return
