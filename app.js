// Pulling the data from a file via local web hosting.
// Performing data manipulation within a function.

d3.json(
  "https://github.com/BootCampUCD.github.io/samples.json"
).then(function (data) {
  var nameNum = data.names;
  var metaData = data.metadata;
  var samplesData = data.samples;
  console.log("samplesData begin:", samplesData);
  var displayStatus = 0;

  function optionChanged() {
    var dataset = d3.selectAll("#selDataset").property("value"); //retrieves data selected from dropdown
    console.log("testing: ", dataset);
    //Types a message if "undefined" is selected from drop down list.
    if (dataset == "undefined") {
      var demoArea = d3
        .select("#demoInfodata")
        .text(
          "Each Test Subject's Demographic Data Will Display Here After Choosing a Subject ID Number Above."
        );
      var sampleDisplayArea = document.getElementById("bar"); // identifies display area tags
      var sampleDataDisplay = d3.select("#bar").text(""); //clears previous data displays.
    }
    // console.log("Selected from Dropdown list", dataset);
    var displayArea = document.getElementById("demoInfodata"); // identifies display area tags
    console.log("displayArea--", displayArea);
    console.log("dataset value:", dataset);
    console.log(displayArea);
    console.log("metaData length:", metaData.length);

    for (i = 0; i < metaData.length; i++) {
      // console.log(i + ",0", metaData[i]);
      if (metaData[i].id == dataset) {
        var age = metaData[i].age;
        var bbtype = metaData[i].bbtype;
        var ethnicity = metaData[i].ethnicity;
        var gender = metaData[i].gender;
        var id = metaData[i].id;
        var location = metaData[i].location;
        var wfreq = metaData[i].wfreq;
        console.log("demoArea b4", demoArea);
        // var demoArea = d3.select("panel-body").select("#demoInfodata").html("");
        var demoArea = d3
          .select("#demoInfodata")
          // .append("h5") //If this is eliminated it will automatically clear the previous list when the next option is selected.  So simple.
          .text("id:" + id);
        var demoArea = d3
          .select("#demoInfodata")
          .append("h5")
          .text("ethnicity: " + ethnicity);
        var demoArea = d3
          .select("#demoInfodata")
          .append("h5")
          .text("gender: " + gender);
        var demoArea = d3
          .select("#demoInfodata")
          .append("h5")
          .text("age: " + age);
        var demoArea = d3
          .select("#demoInfodata")
          .append("h5")
          .text("location: " + location);
        var demoArea = d3
          .select("#demoInfodata")
          .append("h5")
          .text("bbtype: " + bbtype);
        var demoArea = d3
          .select("#demoInfodata")
          .append("h5")
          .text("wfreq: " + wfreq);
      }
    }
    chartSamplesData(samplesData);
  }
  function chartSamplesData(item) {
    var dataset = d3.selectAll("#selDataset").property("value"); //retrieves data selected from dropdown
    console.log("testing2-", dataset);
    var sampleDisplayArea = document.getElementById("bar"); // identifies display area tags
    for (i = 0; i < samplesData.length; i++) {
      if (samplesData[i].id == dataset) {
        console.log("samplesData: #" + i, samplesData[i].id);
        console.log("samplesData array: #" + i, samplesData[i]);
        var id = samplesData[i].id;
        var otu_ids = samplesData[i].otu_ids;
        var otu_labels = samplesData[i].otu_labels;
        var sample_values = samplesData[i].sample_values;
        console.log("id: ", id);
        console.log("otu_id b4", otu_ids);
        console.log("otu_labels b4", otu_labels);
        console.log("sample_values b4", sample_values);
        // var sampleDataDisplay = d3
        //   .select("#bar")
        //   // .append("div") //If this is eliminated it will automatically clear the previous list when the next option is selected. So simple.
        //   .text("id: " + id);
        // console.log("ID length:", id.length);
        // var sampleDataDisplay = d3
        //   .select("#bar")
        //   .append("div")
        //   .text("otu_ids: " + otu_ids);
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

      var trace1 = {
        type: "bar",
        orientation: "h",
        x: sample_values,
        y: otu_ids,
        line: { color: "#17BECF" },
      };
      console.log(trace1);

      var trace2 = {
        type: "scatter",
        mode: "markers",
        x: sample_values,
        y: otu_ids,
        line: { color: "#17BECF" },
      };
      console.log(trace2);

      var data1 = [trace1];
      var data2 = [trace2];
      console.log(data);
      var layout1 = {
        height: 600,
        witdth: 600,
        legend: {
          showlegend: true,
          side: "left",
        },
        title: " Test Subject's Samples Data",
        xaxis: {},
        yaxis: {},
      };
      console.log("Layout1:", layout1);

      var layout2 = {
        height: 600,
        witdth: 600,
        legend: {
          showlegend: true,
          side: "left",
        },
        title: " Test Subject's Samples Data",
        xaxis: {},
        yaxis: {},
      };
      console.log("Layout2:", layout2);
      Plotly.newPlot("bar", data1, layout1, { scrollZoom: true });
      Plotly.newPlot("bubble", data2, layout2);
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
  //Assign List of Test Subject ID.
  var dropDownId = nameNum;
  //cycles through the array and adds each key to the drop down list.
  dropDownId.forEach(addToDropdown);
});
