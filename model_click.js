
// https://newbedev.com/javascript-how-to-pass-parameter-to-event-listener-javascript-code-example


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
    });

}
