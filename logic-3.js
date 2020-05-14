// retrieves data from web site via URL request and process via d3.
// var queryUrl =
//   "https://earthquake.usgs.gov/fdsnws/event/1/query?format=geojson&starttime=2014-01-01&endtime=2014-01-02&maxlongitude=-69.52148437&minlongitude=-123.83789062&maxlatitude=48.74894534&minlatitude=25.16517337";

// d3.json(queryUrl, function (data) {
//   createFeatures1(data.features);
//   console.log(data);
// });
// * * * THIS CODE ABOVE WORKS * * *

//  * * * LINK TO DATABASE * * *

const { Client } = require("pg");
var connectionString = "postgres://postgres:postgres@localhost:5432/project_2";
const client = new Client({
  connectionString: connectionString,
});

// * * * END OF LINK TO DATABASE * * *
//
// dataFile1 = "https://bootcampucd.github.io/us-zip-code-latitude-and-longitude.geojson";
dataFile2 = "https://bootcampucd.github.io/P3-State-Lat-Long.geojson";

d3.json(dataFile2, function (d) {
  createFeatures1(d);
  console.log("line 20: " + d);
});

// * * * GLOBAL VALUES * * *

//change the default marker to a red marker consistent with emergency/earthquake
L.Icon.Default.prototype.options.iconUrl = "https://bootcampucd.github.io/population.png";
L.Icon.Default.prototype.options.iconSize = [30, 30];

//temporary changing the marker, however, couldn't get it to work like i wanted.
var myIcon1 = L.Icon.extend({
  options: {
    iconUrl: "https://bootcampucd.github.io/jobs1.jpg",
    iconSize: [35, 35],
    shadowSize: [50, 64],
    iconAnchor: [0, 0],
    popupAnchor: [-5, -25],
  },
});

//tying to be creative and funny here...couldn't get it to work
var myIcon2 = L.Icon.extend({
  iconUrl: "https://bootcampucd.github.io/face.jpg",
});

// Base map layers.
var streetmap = L.tileLayer(
  "https://api.tiles.mapbox.com/v4/{id}/{z}/{x}/{y}.png?access_token={accessToken}",
  {
    attribution:
      'Map data &copy; <a href="https://www.openstreetmap.org/">OpenStreetMap</a> contributors, <a href="https://creativecommons.org/licenses/by-sa/2.0/">CC-BY-SA</a>, Imagery © <a href="https://www.mapbox.com/">Mapbox</a>',
    maxZoom: 18,
    id: "mapbox.streets",
    accessToken: API_KEY,
  }
);

var skyView = L.tileLayer(
  "https://api.tiles.mapbox.com/v4/{id}/{z}/{x}/{y}.png?access_token={accessToken}",
  {
    attribution:
      'Map data &copy; <a href="https://www.openstreetmap.org/">OpenStreetMap</a> contributors, <a href="https://creativecommons.org/licenses/by-sa/2.0/">CC-BY-SA</a>, Imagery © <a href="https://www.mapbox.com/">Mapbox</a>',
    maxZoom: 18,
    id: "mapbox.satellite",
    accessToken: API_KEY,
  }
);
// Default map displayed on the web page
var myMap = L.map("map", {
  center: [37.09, -95.71],
  zoom: 3,
  layers: streetmap, //this map shows first with circles for population by State
});
// baseMaps
var baseMaps = {
  // "Satellite View": skyView,
  "Street Map": streetmap,
};

//* * * trying to get an action when "clicking" the marker * * *
var oldLayer = "";
function clickx(d) {
  var layer = d.target;
  alert("Marker clicked!");
  layer.setIcon((layer.options.icon = myIcon2));
}

// * * * END OF GLOBAL VALUES * * *

function createFeatures1(earthquakeData) {
  function eachLayersFeature1(feature1, layer1) {
    // add the details of the population when clicking the marker.
    // * * * This is Statistics for the Population layer icons * * *
    layer1.bindPopup(
      "<h3>" +
        feature1.properties.State +
        " (" +
        feature1.properties.Abbrev +
        ") " +
        " (" +
        feature1.properties.Initials +
        ") " +
        "</h3><hr><p>" +
        "1990 Population: " +
        feature1.properties.Pop_1990 +
        "<BR />" +
        "2000 Population: " +
        feature1.properties.Pop_2000 +
        "<BR />" +
        "2010 Population: " +
        feature1.properties.Pop_2010 +
        "<BR />" +
        "2018 Population: " +
        feature1.properties.Pop_2018 +
        "<BR />" +
        "</h3><hr><p>" +
        " Lattitude: " +
        feature1.geometry.coordinates[1] +
        " Lattitude: " +
        feature1.geometry.coordinates[0]
    );

    var latlng1 = [
      feature1.geometry.coordinates[1],
      feature1.geometry.coordinates[0],
    ];
    console.log("latlng1: " + latlng1);

    // Amplified the circle by multiplying the population by a factor to proportionally adjust the circle diameter for visual purposes only.
    var radius1 = feature1.properties.UR_Pop * 0.0025;
    var circleStyle1 = {
      fillOpacity: 1,
      color: "red",
      // fillColor: "red",
      radius: radius1,
    };
    // console.log("radiusx: " + radiusx);
    L.circle(latlng1, circleStyle1).addTo(myMap);
    var markerX2 = L.marker(latlng1, { icon: myIcon1 }); //.on("click", { clickx });
    markerX2.on("click", function (e) {
      if (myMap.hasLayer(markerX2)) {
        markerX2.remove(myMap);
      } else {
        markerX2.addTo(myMap);
      }
    });
    // L.marker(latlng1, { icon: myIcon2 }).addTo(myMap);
  }

  // * * * END   This is Statistics for the Population layer icons * * *

  function eachLayersFeature2(feature2, layer2) {
    // * * * left this description out since we couldn't get the circles that made sense when displayed * * *
    // This is Popup for the new Overlay map

    layer2.bindPopup(
      "<h3>" +
        feature2.properties.State +
        " (" +
        feature2.properties.Abbrev +
        ") " +
        " (" +
        feature2.properties.Initials +
        ") " +
        "</h3><hr><p>" +
        "Unemployment Rate: " +
        feature2.properties.UR * 100 +
        "%" +
        "<BR>" +
        "Unemployment Population: " +
        parseInt(feature2.properties.UR * feature2.properties.UR_Pop)
    );
    var latlng2 = [
      feature2.geometry.coordinates[1],
      feature2.geometry.coordinates[0],
    ];
    console.log("latlng2: " + latlng2);

    // Amplified the circle by multiplying the population by a factor to proportionally adjust the circle diameter for visual purposes only.
    // var radius2 = feature2.properties.Pop_2018 * 0.002;
    // console.log("Population: " + feature2.properties.Pop_2018);
    // var circleStyle2 = {
    //   fillOpacity: 100,
    //   color: "Blue",
    //   fillColor: "black",
    //   radius: radius2,
    // };
    // console.log("radiusx: " + radiusx);

    //* * * Different ways to draw a circle with different attributes * * *
    // layer2.bindPopup(L.circleMarker(latlngx, { radius: radiusx }).addTo(myMap));
    // L.circleMarker(L.latLng(latlngx), {
    //   radius: radiusx,
    // }).addTo(myMap);
    // layer2.bindPopup(L.marker(latlngx, { icon: myIcon }).addTo(myMap));
    var markerX2 = L.marker(latlng2, { icon: myIcon1 }); //.on("click", { clickx });
    markerX2.on("click", function (e) {
      if (myMap.hasLayer(markerX2)) {
        markerX2.remove(myMap);
      } else {
        markerX2.addTo(myMap);
      }
    });
    // L.circle(latlng2, circleStyle2).addTo(myMap).bindPopup({ myIcon2 });
  }

  function eachLayersFeature3(feature3, layer3) {
    // * * * left this description out since we couldn't get the circles that made sense when displayed * * *

    // layer2.bindPopup(
    //   "<h3>" +
    //     feature2.properties.place +
    //     "</h3><hr><p>" +
    //     new Date(feature2.properties.time) +
    //     "</h3><hr><p>" +
    //     " Magnitude: " +
    //     feature2.properties.mag +
    //     "</h3><hr><p>" +
    //     " Lattitude: " +
    //     feature2.geometry.coordinates[0] +
    //     " Lattitude: " +
    //     feature2.geometry.coordinates[1]
    // );
    var latlng3 = [
      feature3.geometry.coordinates[1],
      feature3.geometry.coordinates[0],
    ];
    console.log("latlng3: " + latlng3);

    // Amplified the circle by multiplying the population by a factor to proportionally adjust the circle diameter for visual purposes only.
    // var radius3 = feature3.properties.Pop_2018 * 0.002;
    // console.log("Population: " + feature3.properties.Pop_2018);
    // var circleStyle3 = {
    //   fillOpacity: 0.25,
    //   color: "green",
    //   fillColor: "orange",
    //   radius: radius3,
    // };
    // console.log("radiusx: " + radiusx);

    //* * * Different ways to draw a circle with different attributes * * *
    // layer2.bindPopup(L.circleMarker(latlngx, { radius: radiusx }).addTo(myMap));
    // L.circleMarker(L.latLng(latlngx), {
    //   radius: radiusx,
    // }).addTo(myMap);
    // layer2.bindPopup(L.marker(latlngx, { icon: myIcon }).addTo(myMap));
    // L.marker(latlngx, { icon: myIcon2 }).addTo(myMap).on("click", { clickx });
    // L.circle(latlng3, circleStyle3).addTo(myMap).bindPopup({ myIcon1 });
  }

  // built in "for" loop cycling through the d3.json array of data.
  var earthquakes1 = L.geoJSON(earthquakeData, {
    onEachFeature: eachLayersFeature1,
  });

  // var markerX2 = L.marker(latlng2, { icon: myIcon1 }); //.on("click", { clickx });
  // markerX2.on("click", function (e) {
  //   if (myMap.hasLayer(markerX2)) {
  //     markerX2.remove(myMap);
  //   } else {
  //     markerX2.addTo(myMap);
  //   }
  // });

  var earthquakes2 = L.geoJSON(earthquakeData, {
    onEachFeature: eachLayersFeature2,
  });

  var earthquakes3 = L.geoJSON(earthquakeData, {
    onEachFeature: eachLayersFeature3,
  });

  console.log("eq3: " + earthquakes3);
  // Sending our earthquakes layer to the createMap1 function
  createMap1(earthquakes1, earthquakes2, earthquakes3);
  // createMap2(earthquakes2);
}
//* * * END OF FUNCTION "createFeature1" * * *

function createMap1(marker1, marker2, marker3) {
  // created an overlay for specific data display.
  var overlayMaps = {
    "Population by State": marker1,
    "Unemployment Rate": marker2,
    // Title: marker2,
    // Magnitude: nullx, // we couldn't isolate the second layer to for circles, so added them to the base terrain map.
  };

  // Adding a menu with toggle for the two (2) different layers.
  L.control.layers(baseMaps, overlayMaps, { collapsed: false }).addTo(myMap);

  // var overlayMapsY = {
  //   "New overlay map": marker3,
  // Title: marker2,
  // Magnitude: nullx, // we couldn't isolate the second layer to for circles, so added them to the base terrain map.
  // };
  // Adding a menu with toggle for the two (2) different layers.
  // L.control.layers(baseMaps, overlayMapsY, { collapsed: false }).addTo(myMap);
  //* * * Trying to add a legend * * *
  // var legendX = L.control
  //   .layers(baseMaps, overlayMapsY, {
  //     collapsed: false,
  //     position: "bottomright",
  //   })
  //   .addTo(myMap);
  // legendX.onAdd = function (map) {
  //   var div = L.DomUtil.create("div", "info legend"),
  //     infox = [1, 2, 3, 4],
  //     labels = ["a", "b", "c", "d"],
  //     innerHTML = "<div><b>Legend</b></div>";
  //   return div;
  // };
  // legendX.addTo(myMap);
  //* * * END OF LEGEND ATTEMPT * * *
}
