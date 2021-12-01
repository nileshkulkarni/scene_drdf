
function enableInteraction() {
    console.log('Hi')
    var x = document.getElementById('blink');
    var start_interaction = true
    if (x.hasAttribute('no-blink')){
        x.removeAttribute('no-blink')
        start_interaction = false
    }
    else{
        var att = document.createAttribute("no-blink");
        x.setAttributeNode(att);
        start_interaction = true
    }
    if (true){
        var x = document.getElementsByTagName('model-viewer');
        console.log(x)
        console.log('button ')
        console.log(x.length)
        Array.from(x).forEach((el) => {
            // Do stuff here
            if (start_interaction) {
                var att = document.createAttribute("camera-controls");
                el.setAttributeNode(att);
            }
            else{
                el.removeAttribute("camera-controls");
            }

            
        });
    }
}

