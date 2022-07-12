import json
import os
import os.path as osp
import pdb

import dominate
import dominate.tags as tags
import imageio
from yattag import Doc, indent

from . import base_html as base_html_utils

onclick_str = 'toggle(this, "{}", "{}")'

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
            console.log(filename)
            console.log(basename_split)
            elm.src = (filename === basename_split) ? trg_img : base_img;
         };
      });
   """


class HTMLWriter(base_html_utils.SingleHTMLWriter):
    def __init__(self, opts):
        super().__init__(opts)

        return

    def add_data(self, data_dict):
        self.data_dict = data_dict
        return

    def add_image(self, tag, img_path, overlay_path=None):
        with tags.div() as img_div:
            if tag is not None:
                p = tags.p(f"{tag}")
                tags.br()
            img_tag = tags.img()
            # img_tag['data-src'] = img_path
            img_tag["src"] = img_path
            img_tag["height"] = "400px"
            img_path_name = osp.basename(img_path)
            if overlay_path is not None:
                # img_tag[
                #     'onclick'
                # ] = "this.src=this.src.substring('{}') ? '{}' : '{}';".format(
                #     img_path_name, overlay_path, img_path
                # )
                img_tag["onclick"] = onclick_str.format(img_path, overlay_path)
                # img_tag['onclick'] = "this.src='{}'".format(overlay_path)
                img_tag["onmouseout"] = f"this.src='{img_path}'"

        return img_div

    def get_save_dir(self, step):
        current_dir = osp.join(self.result_dir, f"{step:08d}")
        os.makedirs(current_dir, exist_ok=True)
        return current_dir

    def dump(self, step, web_dir):

        # current_dir = osp.join(self.result_dir, '{:08d}'.format(step))
        current_dir = self.get_save_dir(step)
        # html_file = osp.join(current_dir, 'data.html')
        doc = dominate.document()

        # with doc.head:
        #     tags.script(
        #         type='text/javascript',
        #         src=
        #         "https://cdn.jsdelivr.net/npm/vanilla-lazyload@12.1.0/dist/lazyload.min.js"
        #     )
        #     tags.script(
        #         src=
        #         "https://ajax.googleapis.com/ajax/libs/jquery/2.1.3/jquery.min.js"
        #     )
        #     tags.style(
        #         """h1 {text-align: center;}
        #                 p {text-align: center;}
        #                 div {text-align: center;}"""
        #  )
        n_items = len(self.data_dict)

        # with doc.head:

        with tags.table(style="width:100%", border="1") as table1:
            for i in range(n_items):
                index = self.data_dict[i]["ind"]
                img_rel_paths = {}
                with tags.tr():
                    with tags.td():
                        tags.p("{}".format("step"))
                        tags.br()
                        tags.p(f"{step}")

                    with tags.td():
                        tags.p("{}".format("ind"))
                        tags.br()
                        tags.p(f"{index}")

                    keys = ["image"]

                    for key in keys:
                        img_path = osp.join(current_dir, f"{index}_{key}.png")
                        imageio.imsave(img_path, self.data_dict[i][key])
                        img_path_rel = osp.relpath(img_path, web_dir)
                        img_rel_paths[key] = img_path_rel
                        with tags.td() as td:
                            img_div = self.add_image(key, img_path_rel)
                            td.add(img_div)

                    # extra_depth_vis = ['depth_img', 'normal_img']
                    # for key in keys:
                    #     if key in self.data_dict[i].keys():

                    keys = [
                        "gt_img",
                        "gt_depth",
                        "gt_normal",
                        "validity",
                        "points_proj",
                        "depth_img",
                        "depth_pred",
                        "normal_img",
                        "normal_img_nv",
                        "depth_ray_gt",
                        "pred_img",
                        "pred_depth",
                        "pred_normal",
                        "depth_gt_0",
                        "depth_gt_1",
                        "depth_pred_0",
                        "depth_pred_1",
                    ]

                    for key in keys:
                        if key not in self.data_dict[i].keys():
                            continue

                        img_path = osp.join(current_dir, f"{index}_{key}.png")

                        if "depth" in key:
                            imageio.imwrite(img_path, self.data_dict[i][key])
                        else:
                            imageio.imsave(img_path, self.data_dict[i][key])

                        img_path_rel = osp.relpath(img_path, web_dir)
                        with tags.td() as td:
                            img_div = self.add_image(
                                key, img_path_rel, overlay_path=img_rel_paths["image"]
                            )
                            td.add(img_div)

                        img_rel_paths[key] = img_path_rel

                    keys = ["gt_href_obj", "pred_href_obj"]
                    for key in keys:
                        if key in self.data_dict[i].keys():
                            with tags.td():
                                html_path = self.data_dict[i][key]
                                href_rel_path = osp.relpath(html_path, self.web_dir)
                                with tags.a(href=f"{href_rel_path}"):
                                    tags.p(key)

                    if "loss" in self.data_dict[i].keys():
                        keys = ["loss"]
                        for key in keys:
                            with tags.td():
                                data = self.data_dict[i][key]
                                tags.p(f"{key} : {data}")

                    if "message" in self.data_dict[i].keys():
                        keys = ["message"]
                        for key in keys:
                            with tags.td():
                                data = self.data_dict[i][key]
                                tags.p(f"{data}")

            # tags.script(onclick_script)
            # tags.script("var LazyLoadInstance = new LazyLoad();")
        # div1 = tags.div()
        # div1.add(table)

        div = tags.div()
        # with tags.div() as tr_div:
        div["class"] = "lazyload"
        div["style"] = "width:100%"
        div.add(table1)

        if "distance_slice_gt" in self.data_dict[i].keys():
            dist_keys = ["distance_slice_gt", "distance_slice_pred"]
            with tags.table(style="width:100%", border="1") as table2:
                with tags.tr():
                    with tags.td():
                        key = dist_keys[0]
                        img_path = osp.join(current_dir, f"{index}_{key}.png")
                        imageio.imsave(img_path, self.data_dict[i][key])
                        img_path_rel = osp.relpath(img_path, web_dir)
                        with tags.td() as td:
                            img_div = self.add_image(
                                None,
                                img_path_rel,
                            )
                            td.add(img_div)

            with tags.table(style="width:100%", border="1") as table3:
                with tags.tr():
                    with tags.td():
                        key = dist_keys[1]
                        img_path = osp.join(current_dir, f"{index}_{key}.png")
                        imageio.imsave(img_path, self.data_dict[i][key])
                        img_path_rel = osp.relpath(img_path, web_dir)
                        with tags.td() as td:
                            img_div = self.add_image(
                                None,
                                img_path_rel,
                            )
                            td.add(img_div)

            div.add(table2)
            div.add(table3)
        return div
