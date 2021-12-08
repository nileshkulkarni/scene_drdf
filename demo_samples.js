


var current_page = 1;
var records_per_page = 2;
var paging = 1
IMAGE_LIST_PATH = './assets/extra_samples/img_list.json'
// IMAGE_LIST_PATH = 'assets/extra_samples/img_list.json'
function load_image_list(dataset) {
    fetch(IMAGE_LIST_PATH)
    .then(response => response.text())
    .then(text => main(JSON.parse(text)[dataset]));
}
  

function main(demo_img_list){
    // console.log(demo_img_list)
    paging = new Pagination(demo_img_list, records_per_page)
    // console.log(paging.display(2))
    // console.log(paging.numOfPages)
    init_model_views(records_per_page)
    create_pages_header(paging.numOfPages, 0)
    changePage(0)
    modelEventListeners()
    changePage(1)
}

function create_pages_header(numOfPages, current_page){
    pagin_table = document.getElementById('paging')
    removeAllChildNodes(pagin_table)
    var tr = document.createElement('TR')
    let td = document.createElement('TD')
    td.innerHTML = "<td> <p style='font-size:22px;'> Pages </p></td>"
    tr.appendChild(td)
    for (var i = 0; i < numOfPages; i++){
        if ((i % 15) == 0){
            pagin_table.appendChild(tr)
            var tr = document.createElement('TR')
        }

        let td = document.createElement('TD')
        if (current_page == i){
            button_str = "<button class='button1 paging_button_selected'>" + ("0" + (i+1)).slice(-2); +  "</button>";
        }
        else{
            button_str = "<button class=' button paging_button' onclick='changePage(" + i + ")'>" + ("0" + (i+1)).slice(-2) +  "</button>";
            
        }
        // var button = document.createElement('BUTTON')
        // button.class = "paging_button"
        // button.onClick = "changePage(i)"
        // button.innerHTML = i.toString()
        td.innerHTML = button_str
        tr.appendChild(td)
    }
    pagin_table.appendChild(tr)
    
}

function changePage(page_num){
    // console.log(paging)
    create_pages_header(paging.numOfPages, page_num)    
    var elements = paging.display(page_num)
    console.log(elements)
    create_data_page(elements, records_per_page)
}



function enableInteractionTieControls() {
    var obj = document.getElementById('tie_controls')
    
    if (obj.hasAttribute('tie-interactions')){
        obj.removeAttribute('tie-interactions')
        // obj.textContent = "Tie Controls"
    }
    else{
        var att = document.createAttribute("tie-interactions");
        obj.setAttributeNode(att);
        console.log('Setting attribute')
        // obj.textContent = "UnTie Controls"
    }

}


var modelViewerLoaded=function(obj, evt){
    obj.jumpCameraToGoal()
    console.log('fired load')
}

var modelViewerClick=function(obj, evt){
    
    // console.log(id)
    
    // if evt.source
    // modelViewer = document.getElementById(id)
    // var orbit  = modelViewer.getCameraOrbit();
    // console.log(orbit)
    var obj_tie = document.getElementById('tie_controls')
    tie_controls = false
    if (obj_tie.hasAttribute('tie-interactions')){
        tie_controls = true
    }
    if (evt.detail.source == 'user-interaction'){
        // console.log('Hi click on model viewer')
        var orbit = obj.getCameraOrbit();
        // console.log(evt)
        
        // console.log(obj.id)
        text = obj.id
        const myArray = text.split("_");
        // console.log(myArray)
        var pred_id = myArray[0].concat('_pred')
        var gt_id = myArray[0].concat('_gt')
        // console.log(pred_id)
        // console.log(gt_id)
        gt_obj = document.getElementById(gt_id)
        pred_obj = document.getElementById(pred_id)
        gt_orbit = gt_obj.getCameraOrbit()
        pred_orbit = pred_obj.getCameraOrbit()
        // console.log('tie controls')
        // console.log(tie_controls)
        if (tie_controls){
            if (gt_id == text){
                // console.log('updating the pred model')
                // pred_orbit.theta = orbit.theta
                // pred_orbit.phi = orbit.phi
                pred_obj.cameraOrbit = orbit.toString()
                // console.log(pred_orbit)
                pred_obj.interactionPrompt = 'none'
                pred_obj.jumpCameraToGoal()
            }
            if (pred_id == text){
                // console.log('updating the gt model')
                // gt_orbit.theta = orbit.theta
                // gt_orbit.phi = orbit.phi
                gt_obj.cameraOrbit = orbit.toString()
                // console.log(gt_orbit)
                gt_obj.interactionPromt = 'none'
                gt_obj.jumpCameraToGoal()
            }
        }
        
        
        
        
        
        
        // var att = document.createAttribute("camera-controls");
        // gt_obj.setAttributeNode(att);
        // att = document.createAttribute("camera-controls");
        // pred_obj.setAttributeNode(att);
        // gt_obj.resetInteractionPrompt()
        // pred_obj.resetInteractionPrompt()
        // gt_obj.jumpCameraToGoal()
        
       
        // gt_obj.cameraOrbit =
        // $theta $phi $radius
    }
    
    // pred_obj = document.getElementById()


}


function modelEventListeners(){
    console.log('enabled interactions')
    var x = document.getElementsByTagName('model-viewer');
    Array.from(x).forEach((el) => {
        var  mvid = el.getAttribute("mv-id")
        var id = el.getAttribute("id")
        // console.log(el)
        // console.log('mv id')
        // // console.log(mvid)
        // console.log(id)
        
        // el.addEventListener('camera-change', function(evt) {
        //     modelViewerClick(evt, mvid)
            
        //     // console.log(evt);
        // })
        el.addEventListener('camera-change', modelViewerClick.bind(event, el), 'false')
        el.addEventListener('load', modelViewerLoaded.bind(event, el), 'false')
        // el.addEventListener('model-visibility', modelViewerLoaded.bind(event, el), 'false')
    });

}


function create_single_model_viewer(tagid){
    var model_viewer_html = '<model-viewer alt="" id=div_name loading="eager" camera-orbit="" style="width:500;height:500" src="" seamless-poster shadow-intensity="1" camera-controls interaction-prompt="none"></model-viewer>'
    return model_viewer_html.replace('div_name', tagid)
}

function create_single_model_div(tagid){
    var div = document.createElement('div');
    div.id = tagid
    
    var table = document.createElement("table")
    table.setAttribute("align","center")
    var tr_head = document.createElement('tr')
    var th_img = document.createElement('td')
    th_img.setAttribute('class', 'th_model th_image')
    th_img.innerHTML = "Input Image"
    var th_gt_model = document.createElement('td')
    th_gt_model.setAttribute('class', 'th_model')
    th_gt_model.innerHTML = "GT Model"
    var th_pred_model = document.createElement('td')
    th_pred_model.setAttribute('class', 'th_model')
    th_pred_model.innerHTML = "DRDF Model"
    tr_head.appendChild(th_img)
    tr_head.appendChild(th_gt_model)
    tr_head.appendChild(th_pred_model)


    var tr_data = document.createElement('tr')
    var td_img = document.createElement('td')
    // td_img.setAttribute('width', '500px')
    td_img.setAttribute('style',  "border-right: thin solid; padding-right:60px")
    var td_gt_model = document.createElement('td')
    td_gt_model.setAttribute('style',  "border-right: thin solid;")
    var td_pred_model = document.createElement('td')
    
    td_img.innerHTML =  "<img id="+tagid+"_img "+  " alt='image path not given' src=''></img>"
    td_gt_model.innerHTML  = create_single_model_viewer(tagid+'_gt')
    td_pred_model.innerHTML = create_single_model_viewer(tagid+'_pred')

    tr_data.appendChild(td_img)
    tr_data.appendChild(td_gt_model)
    tr_data.appendChild(td_pred_model)

    table.appendChild(tr_head)
    table.appendChild(tr_data)
    div.appendChild(table)
    div.setAttribute('style', 'display:none')
    return div

}
function init_model_views(numElementsPerPage){
    model_view_table = document.getElementById("model_views")
    for (var i = 0; i < numElementsPerPage; i++){
       let tagname = "demo"+ i
       let div = create_single_model_div(tagname)
       model_view_table.append(div)
    }
}

function setParametersDemoDir(element, tagName){
    console.log(tagName + "_img")
    var img_tag = document.getElementById(tagName + '_img')
    
    img_tag.src=element['img']

    var model_gt_div = document.getElementById(tagName + '_gt')

    console.log(model_gt_div)
    console.log(element)
    model_gt_div.src = element['gt_model']
    model_gt_div.cameraOrbit = element['camera_controls']
 
    var model_pred_div = document.getElementById(tagName + '_pred')
    console.log(model_pred_div)
    model_pred_div.src = element['pred_model']
    model_pred_div.cameraOrbit = element['camera_controls']
    if (model_gt_div.loaded){
        model_gt_div.jumpCameraToGoal()
    }
    if (model_pred_div.loaded){
        model_pred_div.jumpCameraToGoal()
    }   
}
function create_data_page(elements, numElePerPage){
    for (var ex=0; ex<numElePerPage; ex++){
        var tagName = "demo" + ex
        console.log(tagName)
        ex_div = document.getElementById(tagName)
        if (ex < elements.length){
            ex_div.style.display='block'
            setParametersDemoDir(elements[ex], tagName)
        }
        else{
            ex_div.style.display='none'
        }
    }
}
function removeAllChildNodes(parent) {
    while (parent.firstChild) {
        parent.removeChild(parent.firstChild);
    }
}

function Pagination(pageEleArr, numOfEleToDisplayPerPage) {
    this.pageEleArr = pageEleArr;
    this.numOfEleToDisplayPerPage = numOfEleToDisplayPerPage;
    // console.log(pageEleArr)
    this.elementCount = this.pageEleArr.length;
    this.numOfPages = Math.ceil(this.elementCount / this.numOfEleToDisplayPerPage);
    // console.log('Num pages')
    // console.log(this.numOfPages) 
    const pageElementsArr = function (arr, eleDispCount) {
        const arrLen = arr.length;
        const noOfPages = Math.ceil(arrLen / eleDispCount);
        let pageArr = [];
        let perPageArr = [];
        let index = 0;
        let condition = 0;
        let remainingEleInArr = 0;

        for (let i = 0; i < noOfPages; i++) {

            if (i === 0) {
                index = 0;
                condition = eleDispCount;
            }
            for (let j = index; j < condition; j++) {
                perPageArr.push(arr[j]);
            }
            pageArr.push(perPageArr);
            if (i === 0) {
                remainingEleInArr = arrLen - perPageArr.length;
            } else {
                remainingEleInArr = remainingEleInArr - perPageArr.length;
            }

            if (remainingEleInArr > 0) {
                if (remainingEleInArr > eleDispCount) {
                    index = index + eleDispCount;
                    condition = condition + eleDispCount;
                } else {
                    index = index + perPageArr.length;
                    condition = condition + remainingEleInArr;
                }
            }
            perPageArr = [];
        }
        return pageArr;
    }
    this.display = function (pageNo) {
        if (pageNo >= this.numOfPages || pageNo < 0) {
            return -1;
        } else {
            // console.log('Inside else loop in display method');
            // console.log(pageElementsArr(this.pageEleArr, this.numOfEleToDisplayPerPage));
            // console.log(pageElementsArr(this.pageEleArr, this.numOfEleToDisplayPerPage)[pageNo - 1]);
            return pageElementsArr(this.pageEleArr, this.numOfEleToDisplayPerPage)[pageNo];
        }
    }
}


