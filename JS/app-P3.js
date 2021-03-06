// Pulling the data from a file via local web hosting.
// Performing data manipulation within a function.

// var url = "http://127.0.0.1:5501/Python/app.py";

data1 = "http://127.0.0.1:5501/Data/P3-State-Lat-Long.geojson";
// data2 = "http://127.0.0.1:5501/Data/P3-unemployment.json";
// data3 = "http://127.0.0.1:5432/Data/P3-State-Lat-Long.geojson".then();

function commas(x) {
  return x.toString().replace(/\B(?<!\.\d*)(?=(\d{3})+(?!\d))/g, ",");
}

function decimal1(y) {
  return y.toFixed(2);
}

function decimal2(y) {
  return y.toFixed(4);
}
const deathNumY = "";
function adjustX(x) {
  for (i = 0; i < datax.features.length; i++) {
    deathNumY += datax.features[i].properties.Deaths;
  }
}

// * * * This isolates the array of dictionaries into a list of States for the drop down * * *
d3.json(data1).then(function (datax) {
  var displayStatus = 0;
  const states = [datax.features.length];
  const unEmpRate = [datax.features.length];
  const unEmpPop = [datax.features.length];
  const deathRate = [datax.features.length];
  const deathNum = [datax.features.length];
  const deathNum2 = [datax.features.length];
  const deathNumX = [datax.features.length];
  const statePop = [datax.features.length];
  const statePopX = [];
  const percentOfTotal = [datax.features.length];

  for (i = 0; i < datax.features.length; i++) {
    states[i] = datax.features[i].properties.State;
    unEmpRate[i] = datax.features[i].properties.UR;
    unEmpPop[i] = commas(datax.features[i].properties.UR_Pop);
    deathRate[i] = decimal1(datax.features[i].properties.Percent_D * 100);
    deathNum[i] = commas(datax.features[i].properties.Deaths);
    deathNum2[i] = commas(datax.features[i].properties.Deaths * 500);
    statePop[i] = commas(datax.features[i].properties.Population_D);
    percentOfTotal[i] = decimal2(
      datax.features[i].properties.Deaths /
        datax.features[i].properties.Population_D
    );
  }
  // * * * END * * *This isolates the data for display * * *

  function optionChanged() {
    var dataset = d3.selectAll("#selDataset").property("value"); //retrieves data selected from dropdown
    //Types a message if "undefined" is selected from drop down list.
    if (dataset == "undefined") {
      var demoArea = d3
        .select("#demoInfodata")
        .text(
          "Each Year's Population Data Will Display Here After Choosing a Year to Compare."
        );
      var sampleDisplayArea = document.getElementById("bar"); // identifies display area tags
      var sampleDataDisplay = d3.select("#bar").text(""); //clears previous data displays.
    }
    // console.log("Selected from Dropdown list", dataset);
    var displayArea = document.getElementById("demoInfodata"); // identifies display area tags
    // console.log("displayArea--", displayArea);
    for (i = 0; i < datax.features.length; i++) {
      if (datax.features[i].properties.State == dataset) {
        // var state = datax[i].State;
        // var unemp_rate = data.unemp_rate[i];
        // var bbtype = metaData[i].bbtype;
        // var ethnicity = metaData[i].ethnicity;
        // var gender = metaData[i].gender;
        // var id = metaData[i].id;
        // var location = metaData[i].location;
        // var wfreq = metaData[i].wfreq;
        console.log("demoArea b4", demoArea);
        var demoArea = d3
          .select("#demoInfodata")
          // .append("h5") //If this is eliminated it will automatically clear the previous list when the next option is selected.  So simple.
          .text("State: " + states[i]);
        var demoArea = d3
          .select("#demoInfodata")
          .append("h5")
          .text("Total Population: " + statePop[i]);
        var demoArea = d3
          .select("#demoInfodata")
          .append("h5")
          .text("COVID-19 Deaths: " + deathNum[i] + " (" + deathRate[i] + "%)");
        var demoArea = d3
          .select("#demoInfodata")
          .append("h5")
          .text(
            "State vs Nation of COVID-19 Deaths: " +
              " (" +
              percentOfTotal[i] +
              "%)"
          );
      }
    }
    chartSamplesData(datax);
  }
  function chartSamplesData(item) {
    var dataset = d3.selectAll("#selDataset").property("value"); //retrieves data selected from dropdown
    console.log("testing2-", dataset);
    var sampleDisplayArea = document.getElementById("bar"); // identifies display area tags
    for (i = 0; i < datax.features.length; i++) {
      if (datax.features[i].properties.State == dataset) {
        var state = datax.features[i].properties.State;
        var deathNumX = commas(datax.features[i].properties.Deaths);
        var statePopX = commas(datax.features[i].properties.Population_D);
        var sampleDataDisplay = d3
          .select("#bar")
          // .append("div") //If this is eliminated it will automatically clear the previous list when the next option is selected. So simple.
          .text("State: " + state + " COVID-19 Deaths: " + deathNumX);
        // console.log("ID length:", id.length);
        // var sampleDataDisplay = d3
        //   .select("#bar")
        //   .append("div")
        //   .text("COVID-19 Deaths: " + deathNumX);
        // console.log("otu_ids length:", otu_ids.length);
        // var sampleDataDisplay = d3
        //   .select("#bar")
        //   .append("div")
        //   .text("otu_labels: " + otu_labels);
        // var sampleDataDisplay = d3
        //   .select("#bar")
        //   .append("div")
        //   .text("sample_values: " + sample_values);
      }
    }
    ChartFormat();
    // updateChartData();

    function ChartFormat() {
      var sampleDisplayArea = document.getElementById("bar");
      console.log("deathNum: " + deathNum);
      console.log("statePop: " + statePop);
      var trace1 = {
        name: "COVID Deaths Enhanced 500 times for visual realtivity",
        type: "scatter",
        orientation: "v",
        x: states,
        y: deathNum2,
        line: { color: "red" },
        display: "inline-block",
      };
      // console.log(trace1);

      var trace1a = {
        name: "Total States Population",
        type: "bar",
        // mode: "markers",
        x: states,
        y: statePop,
        // xaxis: "x2",
        // yaxis: "y2",
        line: { color: "blue" },
        display: "inline-block",
      };

      // var trace2 = {
      //   type: "scatter",
      //   mode: "markers",
      //   x: sample_values,
      //   y: otu_ids,
      //   xaxis: "x2",
      //   yaxis: "y2",
      //   line: { color: "#17BECF" },
      //   display: "inline-block",
      // };

      var data = [trace1, trace1a];
      // var data2 = [trace2];
      // console.log(data);
      var layout1 = {
        height: 300,
        width: 700,
        showlegend: true,
        legend: {
          text: "titleX",
          side: "left",
        },
        title: "Number of COVID-19 Deaths",
        xaxis: {
          title: {
            text: "States",
          },
        },
        yaxis: {
          title: {
            text: "Population Count",
          },
        },
      };
      console.log("Layout1:", layout1);

      var layout2 = {
        height: 300,
        width: 400,
        showlegend: true,
        legend: {
          side: "left",
        },
        title: " XYZ Subject's Samples Data",
        xaxis2: {
          title: {
            text: "x Axis",
          },
        },
        yaxis2: {
          title: {
            text: "y axis",
          },
        },
      };

      var layout = [layout1, layout2];
      Plotly.newPlot(
        "bar",
        data,
        layout1,
        { displayModeBar: false },
        { scrollZoom: true }
      );
      // Plotly.newPlot(
      //   "scatter",
      //   data2,
      //   layout2,
      //   { displayModeBar: false },
      //   { scrollZoom: true }
      // );
    }

    // function updateChartData() {
    //   chart.data.datasets[0].data = [5, 10, 20, 12];
    //   chart.update();
    // }
  }

  function addToDropdown(item) {
    var select = document.getElementById("selDataset"); //get the choice made by user
    var option = document.createElement("option"); //creates an element tag
    option.text = item; //define the list of data for the dropdown as a string
    select.add(option); //displays the entire list of data for the dropdown in a vertical list
  }
  d3.selectAll("#selDataset").on("change", optionChanged);
  //set displayed value to "undefined" forcing a User choice before printing any data.
  addToDropdown(dropDownId);

  // * * * Assign List of States to the Drop down prompt * * *
  var dropDownId = states;
  //cycles through the array and adds each key to the drop down list.
  dropDownId.forEach(addToDropdown);
  // * * * END * * * Assign List of States to the Drop down prompt * * *
});
