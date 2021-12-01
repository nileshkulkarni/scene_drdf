function showDemo(id) {
    
    console.log('Clicked')
    if (false){
        var x = document.getElementsByClassName('demo');
        console.log(x)
        console.log('button ', id)
        console.log(x.length)
        Array.from(x).forEach((el) => {
            // Do stuff here
            // console.log(el.style.display)
            if (el.style.display === "none") {
                console.log('coming here')
                el.style.display = "block";
            } else {
                el.style.display = "none";
            }
            // el.style.display = "none";
        });
    }
    
    var rank = [1, 2, 3, 4, 5];
    rank.forEach((ind) =>{
        var demoid = "demo".concat(ind)
        var y = document.getElementById(demoid);
        console.log(demoid)
        console.log(y)
        y.style.display = "none";
        // y.style.visibility="hidden"
    });
    
    console.log('here')
    var demoid = "demo".concat(id)
    var y = document.getElementById(demoid);
    console.log(y)
    y.style.display = "block"
    // y.style.visibility="visible"
    // Array.from(y).forEach((el) => {
    //     // Do stuff here
    //     // console.log(el.style.display)
    //     el.style.display = "block";
       
    // }); 
  }
