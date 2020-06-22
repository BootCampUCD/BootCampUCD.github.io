// * * *  load data in from csv file

const tableData = d3.csv("https://BootCampUCD.github.io/Data/data.csv");

// const tableData = data;

// function uniqueDateInOrder(tableData) {
//   let uniqueDate = new Array();
//   let match = false;
//   let count = 0;
//   for (x = 0; x < tableData.length; x++) {
//     for (y = 0; y < uniqueDate.length; y++) {
//       if (tableData[x].datetime == uniqueDate[y]) {
//         match = true;
//       }
//     }
//     count++;
//     if (match == false && count == 1) {
//       uniqueDate.push(tableData[x].datetime);
//     }
//     count = 0;
//     match = false;
//   }
//   uniqueDate.sort(function (a, b) {
//     return a - b;
//   });

//   return uniqueDate;
// }
// var dropDownDate = uniqueDateInOrder(tableData);

var dropDownDate = state;

dropDownDate.forEach(addToDropdown);

function addToDropdown(item) {
  var select = document.getElementById("dropdown");
  var option = document.createElement("option");
  option.text = item;
  select.add(option);
  console.log(item);
}

d3.selectAll("#dropdown").on("change", function () {
  var table = document.getElementById("ufo-table");
  for (var i = table.rows.length - 1; i > 0; i--) {
    table.deleteRow(i);
  }
  let userInput = d3.select("#dropdown").property("value");
  let filteredData = tableData.filter((i) => tableData.state[i] === userInput);
  getDatedata(filteredData);
});

d3.selectAll("#th_date").on("change", function () {
  var table = document.getElementById("ufo-table");
  for (var i = table.rows.length - 1; i > 0; i--) {
    table.deleteRow(i);
  }
  let userInput = d3.select("#th_date").property("value");
  let filteredData = tableData.filter((i) => i.state === userInput);
  getDatedata(filteredData);
});

function getDatedata(filteredData) {
  var dates = filteredData.map((x) => x.datetime);
  var city = filteredData.map((x) => x.city);
  var state = filteredData.map((x) => x.state);
  var country = filteredData.map((x) => x.country);
  var shape = filteredData.map((x) => x.shape);
  var duration = filteredData.map((x) => x.duration);
  var comments = filteredData.map((x) => x.comments);
  buildTable(dates, city, state, country, shape, duration, comments);
}

function buildTable(dates, city, state, country, shape, duration, comments) {
  var table = d3.select("#ufo-table");
  var tbody = table.select("tbody");
  var trow;
  for (var i = 0; i < dates.length; i++) {
    trow = tbody.append("tr");
    trow.append("td").text(dates[i]);
    trow.append("td").text(city[i]);
    trow.append("td").text(state[i]);
    trow.append("td").text(country[i]);
    trow.append("td").text(shape[i]);
    trow.append("td").text(duration[i]);
    trow.append("td").text(comments[i]);
  }
}
