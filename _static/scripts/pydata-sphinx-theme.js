!function(e){var t={};function o(n){if(t[n])return t[n].exports;var r=t[n]={i:n,l:!1,exports:{}};return e[n].call(r.exports,r,r.exports,o),r.l=!0,r.exports}o.m=e,o.c=t,o.d=function(e,t,n){o.o(e,t)||Object.defineProperty(e,t,{enumerable:!0,get:n})},o.r=function(e){"undefined"!=typeof Symbol&&Symbol.toStringTag&&Object.defineProperty(e,Symbol.toStringTag,{value:"Module"}),Object.defineProperty(e,"__esModule",{value:!0})},o.t=function(e,t){if(1&t&&(e=o(e)),8&t)return e;if(4&t&&"object"==typeof e&&e&&e.__esModule)return e;var n=Object.create(null);if(o.r(n),Object.defineProperty(n,"default",{enumerable:!0,value:e}),2&t&&"string"!=typeof e)for(var r in e)o.d(n,r,function(t){return e[t]}.bind(null,r));return n},o.n=function(e){var t=e&&e.__esModule?function(){return e.default}:function(){return e};return o.d(t,"a",t),t},o.o=function(e,t){return Object.prototype.hasOwnProperty.call(e,t)},o.p="",o(o.s=2)}([function(e,t,o){"use strict";function n(e){"loading"!=document.readyState?e():document.addEventListener("DOMContentLoaded",e)}o.d(t,"a",(function(){return n}))},,function(e,t,o){"use strict";o.r(t);var n=o(0),r=(o.p,window.matchMedia("(prefers-color-scheme: dark)"));function c(e){document.documentElement.dataset.theme=r.matches?"dark":"light"}function a(e){"light"!==e&&"dark"!==e&&"auto"!==e&&(console.error(`Got invalid theme mode: ${e}. Resetting to auto.`),e="auto");var t=r.matches?"dark":"light";document.documentElement.dataset.mode=e;var o="auto"==e?t:e;document.documentElement.dataset.theme=o,localStorage.setItem("mode",e),localStorage.setItem("theme",o),console.log(`[PST]: Changed to ${e} mode using the ${o} theme.`),r.onchange="auto"==e?c:""}function d(){const e=document.documentElement.dataset.defaultMode||"auto",t=localStorage.getItem("mode")||e;var o,n,c=r.matches?["auto","light","dark"]:["auto","dark","light"];a(((n=(o=c).indexOf(t)+1)===o.length&&(n=0),o[n]))}var l=()=>{let e=document.querySelectorAll("form.bd-search");return e.length?(1==e.length?e[0]:document.querySelector("div:not(.search-button__search-container) > form.bd-search")).querySelector("input"):void 0},i=()=>{let e=l(),t=document.querySelector(".search-button__wrapper");e===t.querySelector("input")&&t.classList.toggle("show"),document.activeElement===e?e.blur():(e.focus(),e.select(),e.scrollIntoView({block:"center"}))};function s(e){const t=DOCUMENTATION_OPTIONS.pagename+".html",o=e.target.getAttribute("href");let n=o.replace(t,"");return fetch(o,{method:"HEAD"}).then(()=>{location.href=o}).catch(e=>{location.href=n}),!1}var u=document.querySelectorAll(".version-switcher__button");u.length&&fetch(DOCUMENTATION_OPTIONS.theme_switcher_json_url).then(e=>e.json()).then(e=>{const t=DOCUMENTATION_OPTIONS.pagename+".html";u.forEach(e=>{e.dataset.activeVersionName="",e.dataset.activeVersion=""}),e.forEach(e=>{"name"in e||(e.name=e.version);const o=document.createElement("span");o.textContent=""+e.name;const n=document.createElement("a");n.setAttribute("class","list-group-item list-group-item-action py-1"),n.setAttribute("href",`${e.url}${t}`),n.appendChild(o),n.onclick=s,n.dataset.versionName=e.name,n.dataset.version=e.version,document.querySelector(".version-switcher__menu").append(n),"DOCUMENTATION_OPTIONS.version_switcher_version_match"==e.version&&(n.classList.add("active"),u.forEach(t=>{t.innerText=t.dataset.activeVersionName=e.name,t.dataset.activeVersion=e.version}))})}),Object(n.a)((function(){a(document.documentElement.dataset.mode),document.querySelectorAll(".theme-switch-button").forEach(e=>{e.addEventListener("click",d)})})),Object(n.a)((function(){if(!document.querySelector(".bd-docs-nav"))return;var e=document.querySelector("div.bd-sidebar");let t=parseInt(sessionStorage.getItem("sidebar-scroll-top"),10);if(isNaN(t)){var o=document.querySelector(".bd-docs-nav").querySelectorAll(".active");if(o.length>0){var n=o[o.length-1],r=n.getBoundingClientRect().y-e.getBoundingClientRect().y;if(n.getBoundingClientRect().y>.5*window.innerHeight){let t=.25;e.scrollTop=r-e.clientHeight*t,console.log("[PST]: Scrolled sidebar using last active link...")}}}else e.scrollTop=t,console.log("[PST]: Scrolled sidebar using stored browser position...");window.addEventListener("beforeunload",()=>{sessionStorage.setItem("sidebar-scroll-top",e.scrollTop)})})),Object(n.a)((function(){window.addEventListener("activate.bs.scrollspy",(function(){document.querySelectorAll(".bd-toc-nav a").forEach(e=>{e.parentElement.classList.remove("active")});document.querySelectorAll(".bd-toc-nav a.active").forEach(e=>{e.parentElement.classList.add("active")})}))})),Object(n.a)(()=>{(()=>{let e=document.querySelectorAll("form.bd-search");window.navigator.platform.toUpperCase().indexOf("MAC")>=0&&e.forEach(e=>e.querySelector("kbd.kbd-shortcut__modifier").innerText="⌘")})(),window.addEventListener("keydown",e=>{let t=l();(e.ctrlKey||e.metaKey)&&"KeyK"==e.code?(e.preventDefault(),i()):document.activeElement===t&&"Escape"==e.code&&i()},!0),document.querySelectorAll(".search-button__button").forEach(e=>{e.onclick=i});let e=document.querySelector(".search-button__overlay");e&&(e.onclick=i)}),Object(n.a)((function(){new MutationObserver((e,t)=>{e.forEach(e=>{0!==e.addedNodes.length&&void 0!==e.addedNodes[0].data&&-1!=e.addedNodes[0].data.search("Inserted RTD Footer")&&e.addedNodes.forEach(e=>{document.getElementById("rtd-footer-container").append(e)})})}).observe(document.body,{childList:!0})}))}]);