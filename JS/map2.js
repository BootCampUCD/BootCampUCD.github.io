var dataFile = "../Data/P3-State-Lat-Long.geojson";
var mapboxAccessToken = { API_KEY };
var myMap = L.map("map").setView([37.8, -96], 4);

L.tileLayer(
  "https://api.mapbox.com/styles/v1/{id}/tiles/{z}/{x}/{y}?access_token=" +
    mapboxAccessToken,
  {
    id: "mapbox/light-v9",
    // attribution: ,
    tileSize: 512,
    maxZoom: 18,
  }
).addTo(myMap);

L.geoJson(dataFile).addTo(myMap);
