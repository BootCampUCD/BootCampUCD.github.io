// D3 Homework

//-----------------------------------------
// * * * size of display area
var svgWidth = 1200;
var svgHeight = 700;

// * * * boarder margins
var margin = {
  top: 25,
  right: 40,
  bottom: 250,
  left: 150,
};

// * * * display are adjusted by boarder margins
var width = svgWidth - margin.left - margin.right;
var height = svgHeight - margin.top - margin.bottom;

// creat svg item and postiion it within the display boarder area
var svg = d3
  .select(".chart")
  .append("svg")
  .attr("width", svgWidth)
  .attr("height", svgHeight);

var chartGroup = svg
  .append("g")
  .attr("transform", `translate(${margin.left}, ${margin.top})`);

//Generate a random number to alternate the color of circles
var random = parseInt(Math.ceil(Math.random() * 13));

//------------------------------------------

// LOAD THE DATA FROM CSV
// Retrieve data from the CSV file and execute everything below
d3.csv("https://BootCampUCD.github.io/Data/data.csv")
  .then(function (data1, err) {
    if (err) throw err;
    data1;
    //------------------------------------
    //define variable for each of the data points for easy processing of data
    var id = [];
    var state = [];
    var abbr = [];
    var poverty = [];
    var povertyMoe = [];
    var age = [];
    var ageMoe = [];
    var healthcare = [];
    var healthcareHigh = [];
    var healthcareLow = [];
    var income = [];
    var incomeMoe = [];
    var obesity = [];
    var obesityHigh = [];
    var obesityLow = [];
    var smokes = [];
    var smokesHigh = [];
    var smokesLow = [];

    //--------------------------------------
    //populating the individual variables for each State for easy processing
    for (i = 0; i < data1.length; i++) {
      id.push(data1[i].id);
      state.push(data1[i].state);
      abbr.push(data1[i].abbr);
      poverty.push(data1[i].poverty);
      povertyMoe.push(data1[i].povertyMoe);
      age.push(data1[i].age);
      ageMoe.push(data1[i].ageMoe);
      healthcare.push(data1[i].healthcare);
      healthcareHigh.push(data1[i].healthcareHigh);
      healthcareLow.push(data1[i].healthcareLow);
      income.push(data1[i].income);
      incomeMoe.push(data1[i].incomeMoe);
      obesity.push(data1[i].obesity);
      obesityHigh.push(data1[i].obesityHigh);
      obesityLow.push(data1[i].obesityLow);
      smokes.push(data1[i].smokes);
      smokesHigh.push(data1[i].smokesHigh);
      smokesLow.push(data1[i].smokesLow);
    }
    //-------------------------------------
    // trying to integrate User input to select State information to view
    // var inputState = prompt("State?");
    //-------------------------------------

    // defines several colors to change the circles' color upon refresh
    var color = d3
      .scaleQuantize()
      .domain([0, 13]) //d3.max(data1, (d) => `d.${chosenXAxis}`))
      .range([
        "beige", //0
        "blue", //1-transparent/clear
        "#5E4FA2", //2-purple
        "#3288BD", //3-light blue
        "#66C2A5", //4-light teal
        "#ABDDA4", //5-light green
        "#E6F598", //6-faded greenish/yellow
        "#FFFFBF", //7-light faded yellow
        "#FEE08B", //8-light yellow
        "#FDAE61", //9-tan
        "#F46D43", //10-orange?
        "#D53E4F", //11-cherry?
        "#9E0142", //12-maroon?
        "#7f0000", //13-dark maroon?
      ]);
    console.log(random);

    // * * * set intial chart display variable
    var chosenXAxis = "poverty"; //allows changing of data source comparison-not active
    var xLinearScale = d3
      .scaleLinear()
      .domain([0, d3.max(data1, (d, i) => d.poverty) * 3]) //`d.${chosenXAxis}`
      .range([0, width]); //manages the display width on the screen

    // Create y scale function
    var yLinearScale = d3
      .scaleLinear()
      .domain([0, d3.max(data1, (d) => d.healthcare) * 3])
      .range([height, 0]); //manages the display width on the screen

    // Create initial axis functions
    var bottomAxis = d3.axisBottom(xLinearScale);
    var leftAxis = d3.axisLeft(yLinearScale);

    // append x axis line angle
    var xAxis = chartGroup
      .append("g")
      .classed("x-axis", true)
      .attr("transform", `translate(0, ${height})`)
      .call(bottomAxis);

    // append y axis
    chartGroup.append("g").call(leftAxis);

    // plot circles
    var circlesGroup = chartGroup
      .selectAll("circle")
      .data(data1)
      .enter()
      .append("circle")
      .attr("cx", (d, i) => xLinearScale(poverty[i]))
      .attr("cy", (d, i) => yLinearScale(healthcare[i]))
      .attr("r", 15)
      .attr("fill", color(random)) //changes color of circles upon refresh
      .attr("opacity", ".8")
      .style("stroke", "black"); //draws line around the circle for circle identificaiton when the color is a fade/light shaded color

    console.log(color(random));

    //add State initials to each circle - works for some circles but not all
    var circleState = chartGroup
      .selectAll("text")
      .data(data1)
      .enter()
      .append("text")
      .classed("text", true)
      .attr("text-anchor", "middle")
      .attr("alignment-baseline", "middle")
      .attr("x", (d, i) => xLinearScale(poverty[i]))
      .attr("y", (d, i) => yLinearScale(healthcare[i]))
      .text((d, i) => abbr[i])
      .style("color", "blue")
      .style("font-weight", "bold");

    // * * * x-axis labels
    var labelsGroup = chartGroup
      .append("g")
      .attr("transform", `translate(${width / 2}, ${height + 20}) rotate (0)`);

    // * * * x-axis label
    var xaxisLabel = labelsGroup
      .append("text")
      .attr("x", 0)
      .attr("y", 20)
      .style("color", "red")
      .style("font", "25px times")
      .text("Poverty %");

    // * * * y-axis label
    chartGroup
      .append("text")
      .attr("transform", "rotate(-90)")
      .attr("y", 0 - margin.left)
      .attr("x", 0 - height / 2)
      .attr("dy", "3em")
      .classed("axis-text", true)
      .style("font", "25px times")
      .style("color", "blue")
      .text("Healthcare %");

    var circlesGroup = updateToolTip(poverty, circlesGroup);
  })
  .catch(function (error) {
    console.log(error);
  });
//----------------------------------------
// Display data for each State, used in conjunction with "mouseover" function
function updateToolTip(poverty, circlesGroup) {
  var toolTip = d3
    .tip()
    .attr("class", "tooltip")
    .offset([0, 0])
    .html(
      (d) =>
        `State: ${d.state}<br>Healthcare: ${d.healthcare}%<br>Poverty: ${d.poverty}%`
    );

  circlesGroup.call(toolTip);

  circlesGroup
    // When mouse moves over the item data is displayed
    .on("mouseover", function (data) {
      toolTip
        .show(data)
        .style("font", "25px times")
        .style("color", "blue")
        .style("font-weight", "bold");
    })
    // When mouse moves off of the item data stops being displayed
    .on("mouseout", function (data, index) {
      toolTip.hide(data);
    });

  return circlesGroup;
}
//END function: updateToolTip
//----------------------------------------
